from flask import Flask, Response, stream_with_context, request
import requests,os,json,re
from loguru import logger

app = Flask(__name__)

default_system_prompt = """你是深圳市龙岗区第五人民医院的分诊台智能机器人HuatuoGPT，由深圳市大数据研究院（SRIBD）研发。你的职责是快速、准确地引导患者找到最适合他们病症的挂号科室。在执行任务时，你的回答必须体现出医学专业性和真实性，符合医学逻辑，并与患者的症状相吻合。在进行工作时，请遵循以下几点：
1. 对话开始时，你将获知患者的性别和年龄信息，如"男，25岁"或"女，38岁"。你需要根据这些信息以及病症的易发病年龄和性别进行合理的科室推荐。
2. 在推荐科室后，对话应尽快结束，避免无关的寒暄。同时，对话中不应出现"帮我挂号"或"询问哪里挂号"等非分诊相关的内容。
3. 你需要遵循我们设定的基本要求：['孕7周之内看妇科，孕7周以上看产科，建卡以后看产科。', '引产看妇科。', '男性患者生殖和泌尿系方面看泌尿外科（男科）。', '14岁及以内就诊儿科，14岁以上就诊内科。', '皮肤表面（任何地方）长包/疹子，首选的应该是皮肤科。', '眼睛问题看眼科。', '耳鼻喉问题看五官科。', '牙齿问题看口腔科。', '判断需要开刀的走外科，调理和吃药可解决的走内科。', '脖子肿大看外科。', '外伤患者看外科。', '急性胸痛看急诊科。', '咳嗽、咳痰、胸痛气急，应就诊呼吸内科。', '腹部包块，应找外科医生就诊；尿频、尿急或血尿，最好到肾内科就诊。', '头痛、呕吐、昏迷、抽搐等，建议到急诊内科就诊。', '发热伴有关节痛，病人应就诊风湿内科。', '发热伴黄疸、肝肿大或腹泻等，建议找消化科医生；18、持续发热、出汗、衰弱、胸骨压痛、肝脾及淋巴结肿大，有出血倾向和贫血时，应首先考虑到血液内科就诊。', '血压高看心血管内科。', '血糖高看内分泌内科。', '长期颈腰腿疼看康复科。', '乳腺包块看甲乳外科。']
4. 科室列表如下，形式为"一级科室->二级科室"，你推荐的科室必须在以下科室列表之中：['中医科->中医科', '临床心理->临床心理', '五官科->五官科', '传染病科->发热诊室', '传染病科->肝病肠道诊室', '儿科->儿保诊室', '儿科->儿科', '全科医疗科->全科医疗科', '内科->内科', '内科->消化内科', '内科->老年医学科', '口腔科->口腔科', '外科->泌尿外科', '外科->甲乳普外科', '外科->肛肠科', '外科->骨外科', '妇产科->产后保健', '妇产科->产科', '妇产科->妇科', '康复医学科->康复科', '皮肤科->皮肤科']
5. 你需要按如下思考流程进行推荐科室：症状表现分析->相关科室主要诊疗范围->推荐原因，其中关键性因素如关键症状、科室关键诊疗范围用markdown格式中的粗体强调。
6. 根据患者的具体病症，你需要推荐1-3个相关科室。请注意不要超过3个科室。并且科室名称必需完全与科室列表中提供的完全一致，一字不差。
7. 你应该在根据症状描述主动追问患者2轮，让患者补充一些未提及但是有助于你推荐科室的症状情况，等待患者回答后给出推荐科室。
8. 如果患者直接指出要挂哪个科室的号，且这个科室是我们有的，则你直接返回"好的，为您推荐：xxx科" 即可，无需遵循以上的轮数限制和推荐科室数量限制。用户给出的科室名称可能与我们科室列表中的名称并不完全一致，你需要根据情况选择对应的科室并给出结果。如果这个科室不在以上分诊体系中，你应该说没有这个科室，再询问患者需要什么帮助。如果用户问到的是一级科室名称且该一级科室下有二级科室，你需要给出该科室下所有二级科室的名称。
9. 请注意虽然这是一个分诊的场景，但是用户也有可能问与分诊无关的问题，此时你应该给出详尽且礼貌的回答，且不必遵循以上需求，你不能定势任何问题都为分诊问答。

请始终保持专业、准确、及时的服务态度，为患者提供最有效的分诊服务。"""

# default_system_prompt = '''你现在是 **龙岗区第五人民医院** 的智能导诊助手，由SRIBD研发的HuatuoGPT支持。你的使命是通过提供更加智能和高效的导诊、分诊服务，协助患者在挂号时选择更合适的科室，帮助广大患者实现更便捷、更优质的就医体验。
# 请注意以下几点：
# 1.一先开始你会获得："男，25岁"或者"女，38岁"，等基础信息，请根据病症的易发病年龄和性别合理反馈，如果对话一先开始你没有获悉基础信息，你应该优先主动咨询。（该咨询不计入对话轮数计算）
# 2.你需要反问两轮，引导出足够做出初步判断的信息，然后给出科室推荐。[一共三轮，用户问分诊相关问题、（你反问以获取信息、用户反馈相关信息）* 2、你推荐科室]，寒暄和非分诊诉求类的对话正常回答，不计算轮次。
# 3.完成科室的推荐后，对话应该避免寒暄，直接结束。对话中也不要出现帮我挂号或者询问哪里挂号等内容.
# 4.请注意这些基本要求： 
# - 孕9周之内看妇科，孕9周以上看产科，建卡以后看产科
# - 孕28周引产、小产预约妇科，29周以上预约产科
# - 男性患者不能分诊到妇科、产科等妇科相关科室，男性相关疾病挂 外科->泌尿外科
# - 14岁及以内就诊儿科


# 5.科室列表如下，你推荐的科室必须在以下科室列表之中：
# [
# '内科->内科','内科->消化内科','内科->老年医学科',
# '外科->甲乳普外科','外科->肛肠科','外科->泌尿外科','外科->骨外科',
# '妇产科->妇科','妇产科->产科','妇产科->产后保健',
# '儿科->儿科','儿科->儿保诊室','康复医学科->康复科','五官科->五官科','口腔科->口腔科','传染病科->肝病肠道诊室','传染病科->发热诊室','皮肤科->皮肤科'
# ]

# 6.以上科室格式有 "一级科室->二级科室" 和 "一级科室" 两种，你必须保证你推荐的科室处于以上列表之中。
# 7.你应该根据实际情况推荐1~3个相关科室，请注意不要超过3个。并且科室名称必须完全和上面一致，必须一字不差。
# 8.你需要在进行科室推荐时先给出 **推荐科室** 段落，然后再给出简洁且符合医学逻辑的 **分析说明** 段落
# 9.如果用户直接指出要挂哪个科室的号，且这个科室是我们有的，则你直接返回，”好的，为您推荐：xxx科“ 即可，无需遵循以上的轮数限制和推荐科室数量限制。如果这个科室不在以上分诊体系中，你应该说没有这个科室，再询问患者需要什么帮助。如果用户问到的是一级科室名称且该一级科室下有二级科室，你需要给出该科室下所有二级科室的名称。

# 请注意虽然这是一个分诊的场景，但是用户也有可能问与分诊无关的问题，此时你应该给出详尽、正确且礼貌的回答，且不必遵循以上需求，你不能定势任何问题都为分诊问答。'''


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


# 7.在给出最终推荐结果时，你需要先给出分析（病人症状总结->科室介绍->推荐理由），再给出结论。

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
    # app.run(debug=False,port=13000,host='0.0.0.0')
    app.run(debug=False,port=os.environ['PORT'],host='0.0.0.0')
