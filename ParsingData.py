from elasticsearch import Elasticsearch
import pandas as pd


def connect_elasticsearch():
    es = Elasticsearch(
        ['http://es2.danawa.io:9200'],
        http_auth=('elastic', 'ekskdhk1!'),
        timeout=300
    )
    return es

def searchAPI(es, index_name):
    body = {
        "size": 10000,
        "_source": {
            "includes": ['productImageUrl', 'shopCategoryName']
        },
        'query': {
            'match_all': {}
        }
    }
    response = es.search(index=index_name, body=body, scroll='30s')
    return response, response['_scroll_id'], len(response['hits']['hits'])

def generate_dataframe(fetched, response, df_data):
    for i in range(fetched):
        df_data['image'].append(response[i]['_source']['productImageUrl'])
        df_data['category'].append(response[i]['_source']['shopCategoryName'])
    return df_data

if __name__ == '__main__':
    # 엘라스틱 서치 연동
    es_client = connect_elasticsearch()
    data = {'category': [], 'image': []}

    # s-prod 인덱스의 10000개 데이터 검색
    response, scroll_id, fetched = searchAPI(es_client, 's-prod')

    # 초기 10000개의 데이터
    data = generate_dataframe(fetched, response['hits']['hits'], data)

    # 스크롤 속성을 이용하여 모든 데이터 가져오기
    cnt = 0
    while cnt < 490:
        cnt += 1
        if cnt % 50 == 0:
            print(cnt, fetched)
        res = es_client.scroll(scroll_id=scroll_id, scroll='30s')
        fetched = len(res['hits']['hits'])
        data = generate_dataframe(fetched, res['hits']['hits'], data)

    # 데이터 프레임화
    df_data = pd.DataFrame(data)

    # csv파일로 저장장
    df_data.to_csv('data.csv', encoding='utf-8-sig', index=False)

    # 제품 이미지 : productImageUrl
    # 카테고리 : shopCategoryName: '패션잡화>패션소품>머플러'



