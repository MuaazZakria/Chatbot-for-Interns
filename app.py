import streamlit as st
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import warnings
from dotenv import load_dotenv
import os

warnings.filterwarnings("ignore")

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
openai_org_key = os.getenv("OPENAI_ORGANIZATION")



st.title("Chatbot for Interns/Trainees")

# Initialize components (ensure paths and keys are correctly set)
model = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=openai_api_key, openai_organization=openai_org_key)
loaded_vectors = FAISS.load_local("faiss_index", model, allow_dangerous_deserialization=True)

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=openai_api_key,
    organization=openai_org_key
)

prompt_template = """
You are an expert Chat Assistant who helps interns/trainees with their queries.

Given the context, answer the question.

{context}

Question: {question}

INSTRUCTIONS:
- IF the user greets, greet back.
- DO NOT greet with every response.
- IF the context is not similar to the question, respond with 'I don't know the answer'.
- Make the answers short concise and precise.

FORMATTING INSTRUCTION:
- DO NOT add any asterisks in the response.
- Keep the response plain in simple strings.
"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
chain_type_kwargs = {"prompt": PROMPT}

memory = ConversationBufferMemory(
    memory_key="chat_history", output_key="answer", return_messages=True
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    loaded_vectors.as_retriever(search_kwargs={"k": 10}),
    return_source_documents=True,
    memory=memory,
    verbose=False,
    combine_docs_chain_kwargs={"prompt": PROMPT},
)

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4o"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        result = qa_chain.invoke({"question": prompt})
        response = result['answer']
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
