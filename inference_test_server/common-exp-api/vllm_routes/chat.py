import time
import traceback,asyncio
import uuid
from functools import partial
from typing import AsyncIterator

import anyio,os
from fastapi import APIRouter, Depends
from fastapi import HTTPException, Request
from loguru import logger
from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletion,
    ChatCompletionChunk,
)
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_chunk import (
    ChoiceDelta,
    ChoiceDeltaFunctionCall,
    ChoiceDeltaToolCall
)
from openai.types.chat.chat_completion_message import FunctionCall
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
from openai.types.completion_usage import CompletionUsage
from sse_starlette import EventSourceResponse
from vllm.model_executor.guided_decoding import get_guided_decoding_logits_processor
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams

from api.core.vllm_engine import VllmEngine
from api.models import GENERATE_ENGINE
from api.utils.compat import model_dump, model_parse
from api.utils.protocol import Role, ChatCompletionCreateParams
from api.utils.request import (
    check_api_key,
    handle_request,
    get_event_publisher,
)
from utils.kg_tools import rag_dict


default_rag = os.environ.get('DEFAULT_RAG')


chat_router = APIRouter(prefix="/chat")


def get_engine():
    yield GENERATE_ENGINE

def clean_rag_message(request: ChatCompletionCreateParams):
    tag_infos = '''\n\n\n\n\n\n------\n😁 **References:**\n'''
    for turns in request.messages:
        if turns['role'] == 'assistant' and tag_infos in turns['content']:
            kg = turns['content']
            logger.info(f"clean rag message:{kg}")
            turns['content']=""


@chat_router.post("/completions", dependencies=[Depends(check_api_key)])
async def create_chat_completion(
    request: ChatCompletionCreateParams,
    raw_request: Request,
    engine: VllmEngine = Depends(get_engine),
):
    if (not request.messages) or request.messages[-1]["role"] == Role.ASSISTANT:
        raise HTTPException(status_code=400, detail="Invalid request")

    request = await handle_request(request, engine.prompt_adapter.stop)
    request.max_tokens = request.max_tokens or 2048

    params = model_dump(request, exclude={"messages"})
    params['temperature'] = 0.0
    params.update(dict(prompt_or_messages=request.messages, echo=False))
    logger.debug(f"==== request ====\n{params}")

    request_id: str = f"chatcmpl-{str(uuid.uuid4())}"

    query_text = request.messages[-1]['content']

    ref_qa_content = None
    # 没有输入文档 
    logger.info(request.messages[0]['content'])

    rag_setting = params.get('rag_setting')

    sim_question,sim_answer = None,None

    if not rag_setting:
        rag_setting = default_rag
    if rag_setting == 'NO':
        logger.info('no rag in this turns')
    elif ',' not in rag_setting and rag_setting not in rag_dict:
        logger.info(f'error rag settings {rag_setting}')
    # 兼容多个rag的情况
    elif ',' in rag_setting:
        logger.info(f'rag setting {rag_setting} is a list')
        rag_setting_list=rag_setting.split(',')
        for rag in rag_setting_list:
            if rag not in rag_dict:
                logger.info(f'error rag settings {rag}')
            else:
                kg_wrapper = rag_dict[rag]
                logger.info(f'add RAG {rag} kg_wrapper')

                request.messages[-1]['content'], tag_link_paths = kg_wrapper.wrap_question(query_text)
                # 如果已经检索到信息了，不用进下一个rag了
                # if request.messages[-1]['content'] != query_text:
                #     break
                sim_question,sim_answer = kg_wrapper.query_sim_QA(query_text)
                
                if sim_answer:
                    clean_rag_message(request)
                    break
    
    else:

        kg_wrapper = rag_dict[rag_setting]
        logger.info(f'add single RAG {rag_setting} kg_wrapper')
        request.messages[-1]['content'], tag_link_paths = kg_wrapper.wrap_question(query_text)
        sim_question,sim_answer = kg_wrapper.query_sim_QA(query_text)
        clean_rag_message(request)
    
    if sim_answer and request.messages[0]['role'] == 'system':
        logger.info('rag 触发检索问题：Q:{}\n\n A:{}'.format(sim_question,sim_answer))
        # ref_qa_content = 'Q:{}\n\n A:{}'.format(sim_question,sim_answer)

        request.messages[-1]['content'] = "Question:{}\n\n Answer:{}".format(sim_question,sim_answer)

        polish_system = '''
你是文本格式美化大师，接下来我会给你一个QA问答对，你需要在不变更其本身意思的前提下对他的格式进行美化，输出markdown格式的标准化答案（answer）。请注意：
1.如果问题和答案相匹配，不要对答案进行任何语义上的变更，如果问题和答案有逻辑不匹配的地方，可以对答案做适当的调整。
2.格式必须是markdown，并且重点信息要加粗显示。
3.输出的文本要逻辑清晰，层次分明。
4.除了改写后的答案内容，不需要给出任何前置后置的输出。保证文档简洁流程。【注意，只输出改写后的答案】
5.不要参考上下文信息，只根据本次答案进行美化输出。
'''
        request.messages[0]['content'] = polish_system
        logger.info('完成system-prompt功能性替换')
        

    prompt = engine.apply_chat_template(
        request.messages,
        functions=request.functions,
        tools=request.tools,
    )

    if isinstance(prompt, list):
        prompt, token_ids = None, prompt
    else:
        prompt, token_ids = prompt, None

    token_ids = engine.convert_to_inputs(prompt, token_ids, max_tokens=request.max_tokens)
    result_generator = None
    try:
        include = {
            "n",
            "presence_penalty",
            "frequency_penalty",
            "temperature",
            "top_p",
            "repetition_penalty",
            "min_p",
            "best_of",
            "ignore_eos",
            "use_beam_search",
            "skip_special_tokens",
            "spaces_between_special_tokens",
        }
        kwargs = model_dump(request, include=include)
        sampling_params = SamplingParams(
            stop=request.stop or [],
            stop_token_ids=request.stop_token_ids or [],
            max_tokens=request.max_tokens,
            **kwargs,
        )
        lora_request = engine._maybe_get_lora(request.model)
        guided_decode_logits_processor = (
            await get_guided_decoding_logits_processor(
                'lm-format-enforcer',
                request,
                engine.tokenizer,
            )
        )
        if guided_decode_logits_processor:
            sampling_params.logits_processors = sampling_params.logits_processors or []
            sampling_params.logits_processors.append(guided_decode_logits_processor)

        result_generator = engine.model.generate(
            prompt if isinstance(prompt, str) else None,
            sampling_params,
            request_id,
            token_ids,
            lora_request,
        )
    except ValueError as e:
        traceback.print_exc()

    if request.stream:
        
        iterator = create_chat_completion_stream(result_generator, request, request_id, engine,tag_link_paths)
        
        send_chan, recv_chan = anyio.create_memory_object_stream(10)
        return EventSourceResponse(
            recv_chan,
            data_sender_callable=partial(
                get_event_publisher,
                request=raw_request,
                inner_send_chan=send_chan,
                iterator=iterator,
            ),
        )

    else:
        # Non-streaming response
        final_res: RequestOutput = None
        async for res in result_generator:
            if raw_request is not None and await raw_request.is_disconnected():
                await engine.model.abort(request_id)
                return
            final_res = res

        assert final_res is not None
        choices = []
        for output in final_res.outputs:
            
            output.text = output.text.replace("�", "")

            finish_reason = output.finish_reason
            function_call = None
            if request.functions or request.tools:
                try:
                    res, function_call = engine.prompt_adapter.parse_assistant_response(
                        output.text, request.functions, request.tools,
                    )
                    output.text = res
                except Exception as e:
                    traceback.print_exc()
                    logger.warning("Failed to parse tool call")

            if isinstance(function_call, dict) and "arguments" in function_call:
                function_call = FunctionCall(**function_call)
                message = ChatCompletionMessage(
                    role="assistant",
                    content=output.text,
                    function_call=function_call
                )
                finish_reason = "function_call"
            elif isinstance(function_call, dict) and "function" in function_call:
                finish_reason = "tool_calls"
                tool_calls = [model_parse(ChatCompletionMessageToolCall, function_call)]
                message = ChatCompletionMessage(
                    role="assistant",
                    content=output.text,
                    tool_calls=tool_calls,
                )
            else:
                logger.info("---------no stream line--------")
                if tag_link_paths:
                    message = ChatCompletionMessage(role="assistant", content=output.text+'''\n\n\n\n\n\n------\n😁 **References:**\n''' + ','.join(tag_link_paths))
                else:
                    message = ChatCompletionMessage(role="assistant", content=output.text)

            choices.append(
                Choice(
                    index=output.index,
                    message=message,
                    finish_reason=finish_reason,
                )
            )

        num_prompt_tokens = len(final_res.prompt_token_ids)
        num_generated_tokens = sum(len(output.token_ids) for output in final_res.outputs)
        usage = CompletionUsage(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
        )
        clean_rag_message(request)
        return ChatCompletion(
            id=request_id,
            choices=choices,
            created=int(time.time()),
            model=request.model,
            object="chat.completion",
            usage=usage,
        )


async def create_chat_completion_stream(
    generator: AsyncIterator,
    request: ChatCompletionCreateParams,
    request_id: str,
    engine: VllmEngine,
    tag_link_paths
) -> AsyncIterator:
    
    logger.info(f"paths {tag_link_paths}")

    for i in range(request.n):
        # First chunk with role
        choice = ChunkChoice(
            index=i,
            delta=ChoiceDelta(role="assistant", content=""),
            finish_reason=None,
            logprobs=None,
        )
        yield ChatCompletionChunk(
            id=request_id,
            choices=[choice],
            created=int(time.time()),
            model=request.model,
            object="chat.completion.chunk",
        )

        previous_texts = [""] * request.n
        previous_num_tokens = [0] * request.n
        async for res in generator:
            res: RequestOutput
            for output in res.outputs:
                i = output.index
                    
                output.text = output.text.replace("�", "")

                delta_text = output.text[len(previous_texts[i]):]
                previous_texts[i] = output.text
                previous_num_tokens[i] = len(output.token_ids)

                finish_reason = output.finish_reason
                delta = None
                if delta_text == None:
                        delta_text = ''
                if finish_reason == 'stop' and tag_link_paths:
                    
                    delta_text = delta_text +  '''\n\n\n\n\n\n------\n😁 **References:**\n''' + ','.join(tag_link_paths)
                    clean_rag_message(request)

            
                # logger.info(f"模型输出！！！！！: {delta_text}")

                if finish_reason is None or delta_text:
                    delta = ChoiceDelta(content=delta_text)
                elif request.functions or request.tools:
                    call_info = None
                    try:
                        res, call_info = engine.prompt_adapter.parse_assistant_response(
                            output.text, request.functions, request.tools,
                        )
                    except Exception as e:
                        traceback.print_exc()
                        logger.warning("Failed to parse tool call")

                    if isinstance(call_info, dict) and "arguments" in call_info:
                        finish_reason = "function_call"
                        function_call = ChoiceDeltaFunctionCall(**call_info)
                        delta = ChoiceDelta(
                            role="assistant",
                            content=delta_text,
                            function_call=function_call
                        )
                    elif isinstance(call_info, dict) and "function" in call_info:
                        finish_reason = "tool_calls"
                        call_info["index"] = 0
                        tool_calls = [model_parse(ChoiceDeltaToolCall, call_info)]
                        delta = ChoiceDelta(
                            role="assistant",
                            content=delta_text,
                            tool_calls=tool_calls,
                        )
                
                choice = ChunkChoice(
                    index=i,
                    delta=delta or ChoiceDelta(),
                    finish_reason=finish_reason,
                    logprobs=None,
                )
                yield ChatCompletionChunk(
                    id=request_id,
                    choices=[choice],
                    created=int(time.time()),
                    model=request.model,
                    object="chat.completion.chunk",
                )
