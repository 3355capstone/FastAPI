from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import sessionmaker
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.exceptions import NotFittedError
from database import engineconn
from models import PostData

app = FastAPI()

# DB 연결
db_conn = engineconn()
Session = sessionmaker(bind=db_conn.engine)

# 지역별 리스트 정의 및 딕셔너리 생성
seoul = ["서울", "종로", "중구", "용산", "성동", "광진", "동대문", "중랑", "성북", "강북", "도봉", "노원", "은평", "서대문", "마포", "양천", "강서", "구로", "금천", "영등포", "동작", "관악", "서초", "강남", "송파", "강동"]
busan = ["부산", "중구", "서구", "동구", "영도", "부산진", "동래", "남구", "북구", "해운대", "사하", "금정", "강서", "연제", "수영", "사상", "기장"]
daegu = ["대구", "중구", "동구", "서구", "남구", "북구", "수성", "달서", "달성", "군위"]
incheon = ["인천", "중구", "동구", "미추홀", "연수", "남동", "부평", "계양", "서구", "강화", "옹진"]
Gwangju = ["광주", "동구", "서구", "남구", "북구", "광산구"]
Daejeon = ["대전", "동구", "중구", "서구", "유성", "대덕"]
Ulsan = ["울산", "중구", "남구", "동구", "북구", "울주군"]
Gyeonggi = ["경기", "수원", "용인", "고양", "화성", "성남", "부천", "남양주", "안산", "평택", "안양", "시흥", "파주", "김포", "의정부", "광주", "하남", "광명", "군포", "양주", "오산", "이천", "안성", "구리", "의왕", "포천", "양평", "여주", "동두천", "과천", "가평", "연천"]
Gangwon = ["강원", "춘천", "원주", "강릉", "동해", "태백", "속초", "삼척", "홍천", "횡성", "영월", "평창", "정선", "철원", "화천", "양구", "인제", "고성", "양양"]
Chungbuk = ["충북", "청주", "충주", "제천", "보은", "옥천", "영동", "증평", "진천", "괴산", "음성", "단양"]
Chungnam = ["충남", "천안", "공주", "보령", "아산", "서산", "논산", "계룡", "당진", "금산", "부여", "서천", "청양", "홍성", "예산", "태안"]
Jeonbuk = ["전북", "전주", "군산", "익산", "정읍", "남원", "김제", "완주", "진안", "무주", "장수", "임실", "순창", "고창", "부안"]
Jeonnam = ["전남", "목포", "여수", "순천", "나주", "광양", "담양", "곡성", "구례", "고흥", "보성", "화순", "장흥", "강진", "해남", "영암", "무안", "함평", "영광", "장성", "완도", "진도", "신안"]
Gyeongbuk = ["경북", "포항", "경주", "김천", "안동", "구미", "영주", "영천", "상주", "문경", "경산", "의성", "청송", "영양", "영덕", "청도", "고령", "성주", "칠곡", "예천", "봉화", "울진", "울릉"]
Gyeongnam = ["경남", "창원", "진주", "통영", "사천", "김해", "밀양", "거제", "양산", "의령", "함안", "창녕", "고성", "남해", "하동", "산청", "함양", "거창", "합천"]
Jeju = ["제주", "서귀포"]

region_lists = {
    "서울": seoul,
    "부산": busan,
    "대구": daegu,
    "인천": incheon,
    "광주": Gwangju,
    "대전": Daejeon,
    "울산": Ulsan,
    "경기": Gyeonggi,
    "강원": Gangwon,
    "충북": Chungbuk,
    "충남": Chungnam,
    "전북": Jeonbuk,
    "전남": Jeonnam,
    "경북": Gyeongbuk,
    "경남": Gyeongnam,
    "제주": Jeju
}

def map_region_name(input_region):
    for standard_region, region_aliases in region_lists.items():
        if input_region in region_aliases or input_region.startswith(standard_region):
            return standard_region
        for alias in region_aliases:
            if input_region.startswith(alias):
                return standard_region
    raise HTTPException(status_code=400, detail="입력된 지역명이 유효하지 않습니다.")

class InputData(BaseModel):
    region: str
    gender: str
    age: str
    title: str
    message: str

def predict(input_data: InputData, session: Session) -> dict:
    input_region = map_region_name(input_data.region)
    input_gender = input_data.gender
    input_age = input_data.age
    input_title = input_data.title
    input_message = input_data.message

    # 데이터베이스에서 데이터 가져오기
    filtered_data = session.query(PostData).filter(PostData.region.in_(region_lists[input_region])).all()

    if not filtered_data:
        raise HTTPException(status_code=404, detail="입력된 지역과 관련된 데이터가 없습니다.")
    else:
        try:
            # TF-IDF 벡터화
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(
                [f"{data.gender} {data.age} {data.title} {data.message}" for data in filtered_data])

            # 입력된 텍스트를 TF-IDF로 변환
            input_tfidf = vectorizer.transform(
                [f"{input_gender} {input_age} {input_title} {input_message}"])

            # 입력된 텍스트와 데이터 리스트 간의 코사인 유사도 계산
            cosine_similarities = cosine_similarity(input_tfidf, tfidf_matrix).flatten()

            # 유사도가 높은 순서대로 정렬된 인덱스 가져오기
            similar_indices = cosine_similarities.argsort()[::-1]

            # 최대 2개의 유사한 데이터 출력
            result = []
            count = 0
            for i, idx in enumerate(similar_indices):
                if count >= 2:
                    break
                if idx != len(filtered_data) - 1:
                    data = filtered_data[idx]
                    result.append({
                        "rank": count + 1,
                        "data": {
                            "index": data.id,
                            "지역": data.region,
                            "성별": data.gender,
                            "나이대": data.age,
                            "제목": data.title,
                            "동행문구": data.message
                        }
                    })
                    count += 1

            return {"similar_data": result}

        except NotFittedError:
            raise HTTPException(status_code=500, detail="TF-IDF 변환기가 적합되지 않았습니다.")

@app.post("/community")
async def post_predict(input_data: InputData):
    session = Session()
    try:
        return predict(input_data, session)
    finally:
        session.close()

@app.get("/community")
async def get_predict():
    return {"message": "GET 요청은 지원되지 않습니다."}