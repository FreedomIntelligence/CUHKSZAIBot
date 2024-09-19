# CUHKSZAIBot
CUHKSZAIBot, AI Assistant across all Campus


## 1. 数据处理

> /dataProcess

### 元数据组成

- general: 增强性能的通用语料
- school_chat_raw: 由校内工作人员亲自书写或收集的对话
- school_wiki_raw: 前序工作积累的wiki文档
- school_doc_raw: 一些校园指导手册和文档


### 目标数据组成

- general: 增强性能的通用语料
- school_chat: 基于由校内工作人员亲自书写或收集的对话，由AI模型构造的多样问答集。
- school_wiki: 基于整理好的校内wiki，由AI模型构造的多样问答集。
- school_rag: 基于school_wiki,school_chat和general，由AI模型加规则构建的rag多样问答集。

### 数据处理逻辑以及注意事项

- 首先将所有数据统一处理为wiki文档的形式。
  - 保证格式工整，无语法错误。 
- 基于由校内工作人员亲自书写或收集的对话，由AI模型构造多样问答集school_chat。
  - 尽可能多的造复杂的问题，生活化的问题以及多轮有场景的问题，一个数据项造5-10条左右。
- 基于整理好的校内wiki，由AI模型构造多样问答集school_wiki。
  - 一个数据项造3条左右。
- 基于school_wiki, school_chat和general，由AI模型加规则构建rag多样问答集school_rag。
  - 构造时考虑鲁棒性训练(检索错误、无关以及没有检索出信息的情况)。
  - 检索出的文本由特殊标识包围。
  - 构造一些错误的相似但错误的人名或者其他攻击样例，训练模型拒绝回答。

#### 数据处理和构造代码

- 格式化school_doc_raw
```bash
bash 
```

- 扩展school_chat_raw
```bash
bash 
```

- 问答化school_wiki_raw
```bash
bash 
```

- 构造school_rag
```bash
bash 
```

#### 测试集划分以及训练集整合

- 抽10%作为测试集
- 从general中抽取10%作为general_replay


## 2. 模型训练

> /modelTrain

- 训练数据构造
```bash
bash /modelTrain/modelTrain_DataProcess.sh
```

```python
self.data_priority = {
    'general': 16,
    'school_wiki': 4,
    'school_chat': 4,
    'general_replay': 2,
    'school_rag': 2,
}
# 优先级高的先训练
        
self.data_epoch = {
    'general': 1,
    'school_wiki': 1,
    'school_chat': 1,
    'general_replay': 2,
    'school_rag': 2,
}
# 训练时出现的次数
```

- 模型训练
```bash
bash /modelTrain/modelTrain_SingleNode.sh
```


## 3. 模型量化

> /modelQuant

```bash
bash /modelTrain/modelTrain_SingleNode.sh
```


## 4. 模型测评

> /modelEval


## 5. 搭建服务

> /server

#### 目录介绍

- api: 模型API服务拉起(包含RAG逻辑) 
- core: 基于OpenWebUI的前端逻辑 
- db: RAG数据库



## To do

[] 构建数据处理代码
[] 构建模型量化代码
[] 构建测评代码
[] 完成线上部署流程文档化
