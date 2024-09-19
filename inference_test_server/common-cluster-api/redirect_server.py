from flask import Flask, Response, stream_with_context, request
import requests,os,json,io,time
from loguru import logger


current_file_directory = os.path.dirname(os.path.abspath(__file__))

self_host = json.load(io.open(os.path.join(current_file_directory, 'self-host.json'), 'r', encoding='utf-8', errors='ignore'))

text_models = self_host['text-model']

rag_settings = self_host['rag-settings']

app = Flask(__name__)
@app.route('/v1/models', methods=['GET','POST'])
def list_models():

    models = []
    for item in text_models:
        models.append({
            "id": item,
            "object": "model",
            "owned_by": "sribd",
            "created": int(time.time())
        })

    return {"object":"list","data":models}

@app.route('/v1/chat/completions', methods=['GET','POST'])
def stream_forward():
    logger.info(f"source IP: {request.remote_addr}")
    logger.info(f"request header: {request.headers}")
    
    method = request.method
    headers = {key: value for key, value in request.headers if key != 'Host'}
    data = request.data
    

    json_data = json.loads(data)
    target_url = text_models.get(json_data['model'])

    if not target_url:
        return {"error":{"message":"Model not found","type":"invalid_request_error","param":None,"code":None}}
    

    target_url = target_url  + '/chat/completions'
    rag_database = rag_settings.get(json_data['model'])
    if not json_data.get('rag_setting'):
        json_data['rag_setting'] = rag_database if rag_database else 'NO'

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
    # app.run(debug=False,port=8000,host='0.0.0.0')
    app.run(debug=False,port=os.environ['PORT'],host='0.0.0.0')
