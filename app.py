import streamlit as st
import pandas as pd
from langchain.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import CacheBackedEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.callbacks import StdOutCallbackHandler
import transformers
import torch


@st.cache_resource
def load_data():
    file_path = 'cards.csv'
    original_dataset = pd.read_csv(file_path)
    columns = ['name', 'type', 'desc', 'atk', 'def', 'level', 'race', 'attribute', 'archetype']
    final_dataset = original_dataset[columns].copy()
    final_dataset['combined_content'] = final_dataset[['desc', 'type', 'attribute']].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    return final_dataset

final_dataset = load_data()


@st.cache_resource
def prepare_documents():
    loader = DataFrameLoader(final_dataset, page_content_column='combined_content')
    card_info = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, length_function=len)
    return text_splitter.transform_documents(card_info)

cards_documents = prepare_documents()


@st.cache_resource
def configure_embeddings():
    cache_path = LocalFileStore('./cache/')
    embedding_model_id = 'sentence-transformers/all-mpnet-base-v2'
    embedding_model_HF = HuggingFaceEmbeddings(model_name=embedding_model_id)
    embedder = CacheBackedEmbeddings.from_bytes_store(embedding_model_HF, cache_path, namespace=embedding_model_id)
    return FAISS.from_documents(cards_documents, embedder)

vector_store = configure_embeddings()


@st.cache_resource
def configure_model():
    model_id = 'meta-llama/Meta-Llama-3-8B-Instruct'
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model_config = transformers.AutoConfig.from_pretrained(model_id)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    generate_text_pipeline = transformers.pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        do_sample=True,
        temperature=0.1,
        repetition_penalty=1.1,
        return_full_text=True,
        max_new_tokens=256
    )
    llm = HuggingFacePipeline(pipeline=generate_text_pipeline)
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store.as_retriever(),
        callbacks=[StdOutCallbackHandler()],
        return_source_documents=True
    )

qa_with_sources_chain = configure_model()



st.title("Yu-Gi-Oh! Strategy Chatbot")

context_info = "Answer all questions as if you were a Yu-Gi-Oh! expert and only talk about topics related to the trading card game Yu-Gi-Oh!"

query = st.text_input("Enter your question about Yu-Gi-Oh! cards:")

if st.button("Get Answer"):
    if query:
        contextualized_query = context_info + query
        result = qa_with_sources_chain({"query": contextualized_query})
        st.write("Answer:")
        st.write(result['result'])
        st.write("Source Documents:")
        for doc in result['source_documents']:
            st.write(doc.page_content)
    else:
        st.warning("Please enter a query.")