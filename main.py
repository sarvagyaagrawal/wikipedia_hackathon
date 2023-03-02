from typing import Union

from fastapi import FastAPI

from final_script_wiki import *
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "https://wikihack.onrender.com",
    "https://en.wikipedia.org"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/summary_similar_content/{query_str_url}")
def read_item(query_str_url: str):
    query_str=extract_text_from_url(query_str_url)
    # token_df=pd.read_csv("tokenized_wiki_db.csv")
    # union_df=pd.read_csv("union_intersect_wiki_db.csv")
    union_url='https://drive.google.com/file/d/1bG1xodiyinJ4iumLI-gNEBPh6K8YdgTO/view?usp=sharing'
    union_url='https://drive.google.com/uc?id=' + union_url.split('/')[-2]
    
    token_url='https://drive.google.com/file/d/1iObyZ-Ci8AiTVJWU48Nt9JQ8q8qPaa4K/view?usp=share_link'
    token_url='https://drive.google.com/uc?id=' + token_url.split('/')[-2]
    
    token_df=pd.read_csv(token_url)
    union_df=pd.read_csv(union_url)
    obj=summarizer(query_str)
    obj.wrapper()
    
    sim_obj=compute_similarity(obj.final_summary,union_df,token_df)
    sim_obj.get_top_n_articles()
    
    list_json=[]
    for id in sim_obj.matched_ids:
        ind=token_df['id'].loc[lambda x: x==id].index.tolist()
        url=token_df.loc[ind[0],"url"]
        title=token_df.loc[ind[0],"url"]
        score=sim_obj.matched_ids[id]
        list_json.append({"id":id,
                         "url":url,
                         "title":title,
                         "score":score
                         })
    return {
            "relevant ids":list_json,
            "summary": obj.final_summary
           }
    

@app.get("/summary/{query_str_url}")
def read_item(query_str_url: str):
    query_str=extract_text_from_url(query_str_url)
    obj=summarizer(query_str)
    obj.wrapper()
    return {"summary": obj.final_summary}

@app.get("/similar_articles/{query_str_url}")
def read_item(query_str_url: str):
    query_str=extract_text_from_url(query_str_url)
    union_url='https://drive.google.com/file/d/1bG1xodiyinJ4iumLI-gNEBPh6K8YdgTO/view?usp=sharing'
    union_url='https://drive.google.com/uc?id=' + union_url.split('/')[-2]
    
    token_url='https://drive.google.com/file/d/1iObyZ-Ci8AiTVJWU48Nt9JQ8q8qPaa4K/view?usp=share_link'
    token_url='https://drive.google.com/uc?id=' + token_url.split('/')[-2]
    
    token_df=pd.read_csv(token_url)
    union_df=pd.read_csv(union_url)
    sim_obj=compute_similarity(query_str,union_df,token_df)
    sim_obj.get_top_n_articles()
    list_json=[]
    for id in sim_obj.matched_ids:
        ind=token_df['id'].loc[lambda x: x==id].index.tolist()
        url=token_df.loc[ind[0],"url"]
        title=token_df.loc[ind[0],"url"]
        score=sim_obj.matched_ids[id]
        list_json.append({"id":id,
                         "url":url,
                         "title":title,
                         "score":score
                         })
    return {
            "relevant ids":list_json
           }

@app.get("/get_friendliness_score/{query_str_url}")
def read_item(query_str_url):
    query_str=extract_text_from_url(query_str_url)
    # print("query_str:",query_str)
    obj=preprocessData(query_str)
    obj.preprocess_string()
    
    return {"response":friendliness_score(obj.final_str[:200])}