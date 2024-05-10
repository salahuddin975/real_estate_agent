'''
RAG chain using kaggle house peroperty dataset
Dataset link: # https://www.kaggle.com/datasets/polartech/500000-us-homes-data-for-sale-properties
Relevant tutorial link: https://python.langchain.com/v0.1/docs/use_cases/question_answering/quickstart/
Used Qdrant instead of Chroma
'''

from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
import getpass
import os
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate


openai_key='sk-proj-1ZMAlURvUe8g1VPxuAikT3BlbkFJqZbpqkrYMNpuRlfo2Jny'

queries = [
    "As a home buyer with an elementary-school age child, I want to find a house located with high quality school access, safe neighborhoods, park proximity, and good resale value.",
    "As an investor looking for a property in a tourist destination, I want to find locations with high tourist traffic and rental demand to ensure steady income for short-term rentals which can accommodate 5 guests at a time.",
    "As a first-time homebuyer, I want to find a property that fits my budget and meets my needs for space and amenities including a backyard, garage, 2 bedrooms and 1 bathroom.",
    "As a family of two adults and two children who are looking to downsize and reduce the cost of housing, we want to find an apartment or townhouse with 3 bedrooms and 2 bathrooms within 10 miles of our current house so that our kids can continue going to the same school and the parents do not have to change jobs.",
    "As a retired couple looking to simplify our lifestyle, we want to find a manufactured home with two bedrooms and two bathrooms in a 55+ coastal community so that we can enjoy our retirement years in a comfortable setting."
]

# -------------- load dataset ---------------
loader = CSVLoader(file_path='kaggle/600K_USHousingProperties.csv')
data = loader.load()
data = data[:100]

# -------------- 3. Indexing: Store ---------------
key = 'sk-proj-1ZMAlURvUe8g1VPxuAikT3BlbkFJqZbpqkrYMNpuRlfo2Jny'
embedding = OpenAIEmbeddings(openai_api_key=key)

# ------------- 4. Retrieval and Generation: Retrieve -------------
qdrant = Qdrant.from_documents(
    data,
    embedding,
    location=":memory:",                            # Local mode with in-memory storage only
    collection_name="my_documents",
)
retriever = qdrant.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# retrieved_docs = retriever.invoke(query)
# for doc in retrieved_docs:
#     print("retrieved_docs:", doc)

# --------------- 5. Retrieval and Generation: Generate ------------
os.environ["OPENAI_API_KEY"] = openai_key                # getpass.getpass()
llm = ChatOpenAI(model="gpt-3.5-turbo")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

template = """You are an assistant for question-answering tasks. Use the following pieces of context to answer the question at the end.
Always add related property link and address. Always say "thanks for asking!" at the end of the answer.

{context}

Question: {question}

Helpful Answer:"""

custom_rag_prompt = PromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | llm
    | StrOutputParser()
)

for query in queries:
    res = rag_chain.invoke(query)
    print("\nQuery:", query)
    print("Result: ", res)


'''
/Users/smsalahuddinkadir/miniconda3/envs/real_estate_agent/bin/python /Users/smsalahuddinkadir/Documents/Projects/ML/fellowship/real_estate_agent/rag_langchain_with_kaggle_dataset.py 

Query: As a home buyer with an elementary-school age child, I want to find a house located with high quality school access, safe neighborhoods, park proximity, and good resale value.
Result:  Based on your criteria, I recommend considering the property at 1401 Sunnyside Dr, Craig, AK 99921. This single-family home offers 7 bedrooms and 4 bathrooms, making it suitable for a family with an elementary-school age child. The property is located in Craig, AK, with access to Sunnyside Dr and is pending sale, indicating it is in demand. While the listing does not mention school access or park proximity, the property is in a residential area, which typically indicates a safe neighborhood. For more information and to view the property, you can visit the listing here: [1401 Sunnyside Dr, Craig, AK 99921](https://www.zillow.com/homedetails/1401-Sunnyside-Dr-Craig-AK-99921/2102060782_zpid/). Thanks for asking!

Query: As an investor looking for a property in a tourist destination, I want to find locations with high tourist traffic and rental demand to ensure steady income for short-term rentals which can accommodate 5 guests at a time.
Result:  One property that might interest you is the multi-family property located at 203 Chief John Lott St, Petersburg, AK 99833. This property has 4 bedrooms and 4 bathrooms, making it suitable for accommodating 5 guests at a time for short-term rentals. The property is also in a tourist destination area in Petersburg, Alaska, which could attract high tourist traffic and rental demand.

Property Link: [203 Chief John Lott St, Petersburg, AK 99833](https://www.zillow.com/homedetails/203-Chief-John-Lott-St-Petersburg-AK-99833/243013332_zpid/)

Thanks for asking!

Query: As a first-time homebuyer, I want to find a property that fits my budget and meets my needs for space and amenities including a backyard, garage, 2 bedrooms and 1 bathroom.
Result:  Based on your criteria, I recommend checking out the property at 6B Union Bay, Meyers Chuck, AK 99903. It is a single-family home with 2 bedrooms, 1 bathroom, and a spacious land space of 6.91 acres. The price is $195,000, which could fit within your budget as a first-time homebuyer. Additionally, you may have the opportunity to add amenities like a backyard or garage on the generous land space. You can find more information about the property at this link: [6B Union Bay Property Listing](https://www.zillow.com/homedetails/6B-Union-Bay-Meyers-Chuck-AK-99903/2065645308_zpid/). Thanks for asking!

Query: As a family of two adults and two children who are looking to downsize and reduce the cost of housing, we want to find an apartment or townhouse with 3 bedrooms and 2 bathrooms within 10 miles of our current house so that our kids can continue going to the same school and the parents do not have to change jobs.
Result:  I would recommend checking out the property at Mi 2.5 Port Saint Nicholas Rd, Craig, AK 99921. It has 15 bedrooms and 13 bathrooms, which may provide the space you need for your family. The property is also within 10 miles of your current location, allowing your children to continue attending the same school and you to keep your current jobs. You can find more details about the property here: [Mi 2.5 Port Saint Nicholas Rd, Craig, AK 99921](https://www.zillow.com/homedetails/Mi-2-5-Port-Saint-Nicholas-Rd-Craig-AK-99921/2067916130_zpid/). Thanks for asking!

Query: As a retired couple looking to simplify our lifestyle, we want to find a manufactured home with two bedrooms and two bathrooms in a 55+ coastal community so that we can enjoy our retirement years in a comfortable setting.
Result:  Based on your criteria, I recommend checking out the property at L6b3 Coffman Cv, Coffman Cove, AK 99918. It is a single-family home with 2 bedrooms and 2 bathrooms, located in a coastal community. The property offers a comfortable living space and is listed for $399,000. You can find more information about this property at the following link: [L6b3 Coffman Cv Property](https://www.zillow.com/homedetails/L6b3-Coffman-Cv-Coffman-Cove-AK-99918/2068560657_zpid/). Thanks for asking!

Process finished with exit code 0
'''

