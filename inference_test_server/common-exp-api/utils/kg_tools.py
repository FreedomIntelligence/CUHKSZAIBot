import pandas as pd
import io,os,json,re
import random
import gc
import glob
from loguru import logger
from text2vec import SentenceModel,semantic_search
from sentence_transformers import SentenceTransformer
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# os.environ['HF_HUB_ENABLE_HUGGINGFACE_CO_RESOLVE'] = 'false'
# os.environ['TRANSFORMERS_CACHE'] = '/mnt/nvme1n1/huggingface_cache/models'
# os.environ['HUGGINGFACE_HUB_CACHE'] = '/mnt/nvme1n1/huggingface_cache/models'
# os.environ['HF_HOME'] = '/mnt/nvme1n1/huggingface_cache/models'

import torch


def clean_gender_age_prefix(sentence):
    """
    识别并清除句子开头的"性别,年龄"表述。
    
    :param sentence: 输入的句子
    :return: 处理后的句子
    """
    # 定义正则表达式模式，匹配"性别,年龄"的开头
    pattern = r'^(男|女),\d+岁\s*'
    
    # 使用正则表达式替换匹配的部分为空字符串
    cleaned_sentence = re.sub(pattern, '', sentence)
    
    return cleaned_sentence


def is_isolated_word(full_text, data):

    word, start, end = data

    if not re.match(r'^[a-zA-Z]+$', word):
        return True

    # 检查前一个和后一个字符是否是英文字母
    before = start - 1
    after = end
    
    # 检查前后字符是否存在，如果存在且是英文字母，则返回False
    if before >= 0 and re.match(r'[a-zA-Z]', full_text[before]):
        return False
    if after < len(full_text) and re.match(r'[a-zA-Z]', full_text[after]):
        return False
    
    # 如果前后字符不是英文字母或不存在，则返回True
    return True

class DFADictChecker():

    def __init__(self):
        self.keyword_chains = {}
        self.delimit = '\x00'


    def add(self, keyword):
        if not isinstance(keyword, str):
            keyword = keyword.decode('utf-8')
        keyword = keyword.lower()
        chars = keyword.strip()
        if not chars:
            return
        level = self.keyword_chains
        for i in range(len(chars)):
            if chars[i] in level:
                level = level[chars[i]]
            else:
                if not isinstance(level, dict):
                    break
                for j in range(i, len(chars)):
                    level[chars[j]] = {}
                    last_level, last_char = level, chars[j]
                    level = level[chars[j]]
                last_level[last_char] = {self.delimit: 0}
                break
        if i == len(chars) - 1:
            level[self.delimit] = 0

    def parse(self, path):
        with open(path, 'rb') as f:
            for keyword in f:
                self.add(keyword.strip())

    def filter_with_pos(self, message, repl="*"):
        if not isinstance(message, str):
            message = message.decode('utf-8')
        message = message.lower()
        ret = []
        words = []
        start = 0
        while start < len(message):
            level = self.keyword_chains
            step_ins = 0
            for char in message[start:]:
                if char in level:
                    step_ins += 1
                    if self.delimit not in level[char]:
                        level = level[char]
                    else:
                        words.append([message[start:start + step_ins],start,start + step_ins])
                        ret.append(repl * step_ins)
                        level = level[char]
                        # 放开以后，就是只要先完成匹配的一个
                        # start += step_ins - 1
                        # break
                else:
                    ret.append(message[start])
                    break
            else:
                ret.append(message[start])
            start += 1

        return words

    def filter_no_overlap(self, message, repl="*"):
        raw_filter_words = self.filter_with_pos(message=message,repl=repl)
        raw_filter_words.sort(key=lambda x:len(x[0]),reverse=True)
        filter_results = []
        for item in raw_filter_words:
            flag = True
            for candi in filter_results:
                if candi[1] <= item[1] and candi[2] >= item[2]:
                    flag = False
                    break
            if flag:
                filter_results.append(item)
        return filter_results


class KWrapper:

    def parse_sheet(self,sheet_data,except_set_checker):

        sheet_data = sheet_data.fillna('')
        for item_idx in range(len(sheet_data)):
            tag = str(sheet_data['tag'][item_idx]).lower()
            content = str(sheet_data['info'][item_idx]).replace('\\n','\n')
            path = str(sheet_data['path'][item_idx])
            if not tag or not content:
                continue
            try:
                if re.match('^（.*）$',tag) and '|' in tag:
                    all_tags = tag[1:-1].split('|')
                else:
                    all_tags = [tag]

                for i in range(len(all_tags)):
                    except_set_checker.add(all_tags[i])
                    if '内部维护' not in path:
                        self.tag_url_map[all_tags[i]] = f'[{all_tags[i]}]({path})'
                        self.tag_url_map[all_tags[i].lower()] = f'[{all_tags[i]}]({path})'
                    self.tag_content_map[all_tags[i]] = content
          
            except:
                
                except_set_checker.add(tag)
                if '内部维护' not in path:
                    self.tag_url_map[tag] = f'[{all_tags[i]}]({path})'
                    self.tag_url_map[tag.lower()] = f'[{all_tags[i]}]({path})'
                self.tag_content_map[tag] = content.replace('\\n','\n')

    
    def rel_knowledge_concat(self,question,checker):
        name_list = checker.filter_no_overlap(question)
        name_desc_info = ''
        names = []

        tags_path = []

        for idx in range(len(name_list)):
            name_item = name_list[idx]
            is_isolate = is_isolated_word(question,name_item)
            name = name_item[0]
            if name in names or not is_isolate:
                continue
            names.append(name)
            desc = self.tag_content_map[name]
            name_desc_info += (f"{idx+1}.{name}:{desc}\n")

        for name in names:
            tag_path = self.tag_url_map.get(name)
            if tag_path:
                tags_path.append(tag_path)

     

        return name_desc_info,','.join(names), tags_path

    
    def wrap_question(self,question):
        content,keys,tags_path = self.rel_knowledge_concat(question,self.keywords_checker)
        
        if content:

            reduce_question = f'''#检索信息：<{keys}><{content}>#\n\n{question}'''
        else:
            reduce_question = question
        
        return reduce_question,tags_path
    
    def query_sim_QA(self,question):
        content,_,_ = self.rel_knowledge_concat(question,self.keywords_checker)
        
        return question,content


class HuatuoKnowledgeEmbeddingWrapper(KWrapper):

    def __init__(self,base_data_path) -> None:
        # 检索embedding模型
        self.top_k = 1
        self.retrieve_embedding_models = SentenceTransformer(os.environ['EMBEDDING_PATH'])
        # self.retrieve_embedding_models = SentenceModel(os.environ['EMBEDDING_PATH'])#os.environ['EMBEDDING_PATH']
        self.retrieve_corpus,self.key_value_list=self.get_corpus(base_data_path)
        self.retrieve_corpus_embedding_tensor = torch.tensor(self.retrieve_embedding_models.encode(self.retrieve_corpus))
        logger.info(f'rag资料加载完成：{base_data_path}')
        logger.info(f'rag-size：{len(self.retrieve_corpus)}')
    def get_corpus(self, base_data_path):
        with open(base_data_path, 'r', encoding='utf-8') as f:
            # return f.readlines()
            data=json.load(f)
            question_list=[]
            for item in data:
                question_list.append(item[0])
            return question_list,data

        
    def clean_question(self, question):
        return question.rstrip('？').rstrip('?').strip()

    # 查询相关的知识  得到语义相似的问题
    def query_related_context(self, question1):
        # 用户的问题格式可能很乱，比如：体检多少钱? 跟 体检多少钱 相似度不是100% 。所以要清洗。
        question = self.clean_question(question1)
        question_embedding = self.retrieve_embedding_models.encode(question)
        hits=semantic_search(question_embedding,self.retrieve_corpus_embedding_tensor, top_k=self.top_k)
        logger.info("\n\n======================\n\n")
        logger.info("Query:", question1)
        logger.info("\nTop 1 most similar sentences in corpus:")
        relevant_context=[]
        for hit in hits[0]:
            if hit['score']< float(os.environ['EMBEDDING_SCORE_THRESHOLD']): # **挂什么科？相关性是**
                continue
            relevant_context.append(self.retrieve_corpus[hit['corpus_id']].strip())
            logger.info(self.retrieve_corpus[hit['corpus_id']], "(Score: {:.4f})".format(hit['score']))

        logger.info(relevant_context)
        source=""
        return relevant_context,source
    
    # 根据语义相似的问题，得到答案
    def get_answer(self, question:str):
        for item in self.key_value_list:
            if item[0] in question:
                return item[1]
        return ""
    

    def query_sim_QA(self,question):
        forquery_quest = clean_gender_age_prefix(question)

        # 检索到的语义相似的问题 有可能空
        relevant_context,source = self.query_related_context(forquery_quest)
        relevant_answer = self.get_answer(relevant_context)
        logger.info(f"释放资源，包括GPU显存")
        gc.collect()
        torch.cuda.empty_cache()
        return forquery_quest,relevant_answer


    def wrap_question(self, question):
        
        forquery_quest = clean_gender_age_prefix(question)


        # 检索到的语义相似的问题 有可能空
        relevant_context,source = self.query_related_context(forquery_quest) 
        if len(relevant_context) > 0:
            # 语义相似的问题对应的答案
            relevant_answer = self.get_answer(relevant_context)
            reduce_question = f'''#检索信息：<{relevant_answer}>\n\n#问题：<{forquery_quest}>'''  #+{relevant_context[0]} 希望大模型先判断答案跟问题有没有关系 再回答
        else:
            reduce_question = question
            relevant_answer=""
        return reduce_question,[",".join(relevant_context)+"\n"+ relevant_answer] if relevant_answer else [] #最后一项检索到的问题+原始回答 当做source同时用于评估检索质量



class HuatuoKownledgeWrapper(KWrapper):

    def __init__(self,base_data_path) -> None:
        self.kg_ref_path = base_data_path
        self.key_refer = self.consist_group_taginfo(self.kg_ref_path)
        
        
        self.keywords_checker = DFADictChecker()
        self.tag_content_map = {}
        self.tag_url_map = {}
        self.tag_type = {}
        self.parse()
        logger.info(f'rag资料加载完成：{self.kg_ref_path}')
        logger.info(f'rag-word-chain-size：{len(self.keywords_checker.keyword_chains)}')


    def consist_group_taginfo(self,dir_path):
        glob_paths = glob.glob(dir_path + '/*.md')
        reduce_datas = []

        for glb in glob_paths:
            contents = io.open(glb,'r').read()
            all_lines = contents.split('\n')
            tag = all_lines[0]
            info = '\n'.join(all_lines[1:])
            reduce_tag = tag.replace('keys:','').replace('，',',')
            reduce_tag = reduce_tag.replace(',','|')[1:-1]
            reduce_tag = reduce_tag if '|' not in reduce_tag else '（' + reduce_tag + '）'
            knowledge_http_url = 'https://github.com/FreedomIntelligence/longgang_hospitals/blob/main' + glb.split('longgang_hospitals')[1]
            reduce_datas.append({'tag':reduce_tag,'info':info,'path': knowledge_http_url})
        
        return pd.DataFrame(reduce_datas)
    

    def parse(self):
        wait_list = [self.key_refer]
        for key_item in wait_list:
            self.parse_sheet(key_item,self.keywords_checker)


class PhoenixKownledgeWrapper(KWrapper):

    def __init__(self,base_data_path) -> None:
        self.kg_ref_path = base_data_path
        self.key_refer = self.consist_group_taginfo(os.path.join(self.kg_ref_path,'指代信息'))
        self.key_meta = self.consist_group_taginfo(os.path.join(self.kg_ref_path,'元数据'))
        self.key_name = self.consist_group_taginfo(os.path.join(self.kg_ref_path,'教职工人员'))
        self.key_build = self.consist_group_taginfo(os.path.join(self.kg_ref_path,'建筑信息'))
        self.key_landmark = self.consist_group_taginfo(os.path.join(self.kg_ref_path,'地标信息'))
        self.key_subject = self.consist_group_taginfo(os.path.join(self.kg_ref_path,'专业信息'))
        self.key_faculty = self.consist_group_taginfo(os.path.join(self.kg_ref_path,'院系信息'))
        # 内部维护
        self.key_secinfo = self.consist_group_taginfo(os.path.join(self.kg_ref_path,'内部维护'))
        self.surprise = self.consist_group_taginfo(os.path.join(self.kg_ref_path,'彩蛋'))
        
        self.keywords_checker = DFADictChecker()
        self.tag_content_map = {}
        self.tag_url_map = {}
        self.tag_type = {}
        self.parse()
        logger.info(f'rag资料加载完成：{self.kg_ref_path}')
        logger.info(f'rag-word-chain-size：{len(self.keywords_checker.keyword_chains)}')
        
    def consist_group_taginfo(self,dir_path):
        glob_paths = glob.glob(dir_path + '/*.md')
        reduce_datas = []

        for glb in glob_paths:
            contents = io.open(glb,'r').read()
            all_lines = contents.split('\n')
            tag = all_lines[0]
            info = '\n'.join(all_lines[1:])
            reduce_tag = tag.replace('keys:','')
            reduce_tag = reduce_tag.replace(',','|')[1:-1]
            reduce_tag = reduce_tag if '|' not in reduce_tag else '（' + reduce_tag + '）'
            knowledge_http_url = 'https://github.com/FreedomIntelligence/phoenix_cuhksz_knowledge/blob/main' + glb.split('phoenix_cuhksz_knowledge')[1]
            reduce_datas.append({'tag':reduce_tag,'info':info,'path': knowledge_http_url})
        
        return pd.DataFrame(reduce_datas)
    
    def parse(self):
        wait_list = [self.key_refer,self.key_meta,self.key_build,self.key_landmark,self.key_subject,self.key_faculty,self.surprise,self.key_name,self.key_secinfo]
        for key_item in wait_list:
            self.parse_sheet(key_item,self.keywords_checker)
        
    

base_path = os.environ['KG_BASE_PATH']
# base_path = '/mnt/nvme1n1/models/common_flan_phoenix_format-int4_v3/RAG_RES'
rag_dict = {
    # 学校知识
    'school_phoenix' : PhoenixKownledgeWrapper(base_data_path=os.path.join(base_path,'phoenix_cuhksz_knowledge/data_resource')),
    # 龙岗人民医院知识 关键词
    'triage_lgph' : HuatuoKownledgeWrapper(base_data_path=os.path.join(base_path,'longgang_hospitals/data_resource/Longgang_District_People_Hospital')),
    #  龙岗人民医院语义问答对
    'q_a_renmin' : HuatuoKnowledgeEmbeddingWrapper(base_data_path=os.path.join(base_path,'longgang_hospitals/longgang_renmin_hospital/q_a_list_after_gpt4.json')),
    # 龙岗第三人民医院问答对
    'q_a_third_renmin' : HuatuoKnowledgeEmbeddingWrapper(base_data_path=os.path.join(base_path,'longgang_hospitals/longgang_third_renmin_hospital/q_a_list_after_gpt4.json')),
    # 龙岗中心医院问答对
    'q_a_central' : HuatuoKnowledgeEmbeddingWrapper(base_data_path=os.path.join(base_path,'longgang_hospitals/longgang_center_hospital/q_a_list_after_gpt4.json'))


}



if __name__ == '__main__':
    while True:
        question = input('请输入问题：')

        results = rag_dict.get('q_a_renmin').query_sim_QA(question)
        logger.info(results)
    