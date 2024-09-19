from flask import Flask, Response, stream_with_context, request
import requests,os,json,re
from loguru import logger

app = Flask(__name__)

default_system_prompt = """你现在是 **龙岗区域分诊平台** 的智能导诊助手，由SRIBD研发的HuatuoGPT支持。你的使命是通过提供更加智能和高效的导诊、分诊服务，协助患者在挂号时选择更合适的科室，帮助广大患者实现更便捷、更优质的就医体验。
请注意以下几点：
1.一先开始你会获得："男，25岁"或者"女，38岁"，等基础信息，请根据病症的易发病年龄和性别合理反馈，如果对话一先开始你没有获悉基础信息，你应该优先主动咨询。（该咨询不计入对话轮数计算）
2.你需要反问两轮，引导出足够做出初步判断的信息，然后给出科室推荐。[一共三轮，用户问分诊相关问题、（你反问以获取信息、用户反馈相关信息）* 2、你推荐科室]，寒暄和非分诊诉求类的对话正常回答，不计算轮次。
3.完成科室的推荐后，对话应该避免寒暄，直接结束。对话中也不要出现帮我挂号或者询问哪里挂号等内容.
4.请注意这些基本要求： 
- 男性患者不能分诊到妇产科->妇产科，男性相关疾病挂 外科->泌尿外科
- 14岁及以内就诊儿科->儿科

5.科室列表如下，你推荐的科室必须在以下科室列表之中：
['外科->心胸外科', '内科->内分泌科', '内科->心血管内科', '营养科->营养科', '内科->呼吸内科', '外科->甲状腺乳腺外科', '内科->肾内科', '外科->神经外科', '外科->肛肠外科', '生殖健康科->生殖健康科', '五官科->眼科', '康复医学科->康复医学科', '感染性疾病科&发热->感染性疾病科', '五官科->五官科', '妇产科->妇产科', '外科->骨科', '外科->烧伤整形科', '内科->血液内科', '儿科->儿科', '内科->消化内科', '口腔科->口腔科', '外科->普通外科', '疼痛科->疼痛科', '感染性疾病科&发热->发热门诊', '外科->血管外科', '五官科->耳鼻喉科', '肿瘤科->肿瘤科', '外科->泌尿外科', '皮肤科->皮肤科', '内科->神经内科', '内科->风湿免疫科']

6.以上科室格式有 "一级科室->二级科室" 和 "一级科室" 两种，你必须保证你推荐的科室处于以上列表之中。
7.你应该根据实际情况推荐1~3个相关科室，请注意不要超过3个。并且科室名称必须完全和上面一致，必须一字不差。
8.你需要在一段话里简单直接的给出分析说明和科室推荐的内容，不分段落
9.如果用户直接指出要挂哪个科室的号，且这个科室是我们有的，则你直接返回，”好的，为您推荐：xxx科“ 即可，无需遵循以上的轮数限制和推荐科室数量限制。如果这个科室不在以上分诊体系中，你应该说没有这个科室，再询问患者需要什么帮助。如果用户问到的是一级科室名称且该一级科室下有二级科室，你需要给出该科室下所有二级科室的名称。

请注意虽然这是一个分诊的场景，但是用户也有可能问与分诊无关的问题，此时你应该给出详尽、正确且礼貌的回答，且不必遵循以上需求，你不能定势任何问题都为分诊问答。"""

def clean_rag_history(messages):
    filter_messages = []
    tag_infos = '''\n\n\n\n\n\n------\n😁 **References:**\n'''
    filter_messages.append(messages[0])
    for message_idx in range(1,len(messages) - 2,2):
        if tag_infos in messages[message_idx + 1]['content']:
            continue
        else:
            filter_messages.append(messages[message_idx])
            filter_messages.append(messages[message_idx + 1])
    
    filter_messages.append(messages[-1])
    return filter_messages


def departments_query_wrapper(messages):

    add_suffix = '（需反问）'
    respect_suffix = '（用敬语）'
    # 定义正则表达式
    # pattern = re.compile(r'.*(挂(什么|哪个)(科|号|诊室)|去(什么|哪个)科|哪个医生|挂号).{0,4}$')
    pattern = re.compile(r'.*(挂(什么|哪个)(科|号|诊室)|去(什么|哪个)科|看|找哪个医生|挂号).{0,4}$',re.DOTALL)


    def matches_pattern(sentence):
        """
        判断句子是否符合正则表达式
        :param sentence: str, 输入的句子
        :return: bool, 如果句子符合正则表达式返回 True，否则返回 False
        """
        return bool(pattern.match(sentence))
    
    is_match = matches_pattern(messages[1]['content'])
    logger.info(f'is match query more:{is_match}')
    logger.info(messages[1]['content'])
    if len(messages) == 2 and is_match:
        logger.info('add suffix because need ask more questions')
        messages[1]['content'] += add_suffix
    # messages[1]['content'] += respect_suffix


@app.route('/v1/chat/completions', methods=['GET','POST'])
def stream_forward():

    target_url = os.environ['BASE_URL'] + '/v1/chat/completions'

    method = request.method
    headers = {key: value for key, value in request.headers if key != 'Host'}
    data = request.data
    system_prompt = default_system_prompt
    if system_prompt:
        json_data = json.loads(data)
        messages = json_data['messages']
        if messages and messages[0]['role'] != 'system':
            messages = [{'role':'system','content':system_prompt}] + messages
        
        try:
            messages = clean_rag_history(messages)
        except Exception as e:
            logger.error(f'clean_rag_history error:{messages}')
        json_data['messages'] = messages
        json_data['temperature'] = 0.0
        json_data['seed'] = 1024
        json_data['rag_setting'] = 'q_a_third_renmin'
        data = json.dumps(json_data,ensure_ascii=False).encode('utf-8')
    
    # 发送流式请求到目标URL
    req = requests.request(method, target_url, headers=headers, data=data, stream=True, allow_redirects=False)

    # 定义生成器，逐块转发响应数据
    def generate():
        for chunk in req.iter_content(chunk_size=1):
            yield chunk

    # 创建并返回流式响应
    return Response(stream_with_context(generate()), content_type=req.headers['Content-Type'])

if __name__ == '__main__':

    app.run(debug=False,port=os.environ['PORT'],host='0.0.0.0')
