from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=os.getenv("OPENAI_API_KEY"))
large_language_model = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)


class SearchEngine:
    def __init__(self, question, embedding_model=embeddings, llm=large_language_model):
        self.question = question
        self.embedding_model = embedding_model
        self.llm = llm
        self.loader = CSVLoader("urval_kva_2025.csv", encoding="UTF-8")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
        self.vector_store = InMemoryVectorStore(self.embedding_model)

    def load_data(self):
        data = self.loader.load()
        return data

    def split_data(self, data):
        all_splits = self.text_splitter.split_documents(data)
        return all_splits

    def add_documents_to_vector_store(self, all_splits):
        self.vector_store.add_documents(documents=all_splits)

    def retrieve_documents(self):
        retrieved = self.vector_store.search(query=self.question, search_type="mmr")
        return retrieved

    def get_answer(self, retrieved_docs):
        context = "\n\n".join(doc.page_content for doc in retrieved_docs)
        prompt_template = PromptTemplate.from_template("Question: {question}\n\nContext: {context}")
        prompt = prompt_template.invoke({"question": self.question, "context": context})
        answer = self.llm.invoke(prompt)
        return answer.content

    def main(self):
        data = self.load_data()
        all_splits = self.split_data(data)
        self.add_documents_to_vector_store(all_splits)
        retrieved_docs = self.retrieve_documents()
        answer = self.get_answer(retrieved_docs)
        return answer


q = "Vilken KVÅ kod ska jag ange om jag tränat en patient i att klä sig själv?"
s = SearchEngine(q)
a = s.main()
print(a)
