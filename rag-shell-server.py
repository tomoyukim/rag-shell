import time
import os
import logging
from epc.server import EPCServer

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings

from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from bs4 import BeautifulSoup as Soup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch
from langchain_core.runnables import RunnablePassthrough

from langchain_core.messages import HumanMessage


class Config:
    def __init__(self, client):
        self.client = client
    def get_source():
        return self.client.call_sync("get-source")


class RAG:
    def load_web_docs(url, depth=2):
        #TODO: support multiple web pages support
        loader = RecursiveUrlLoader(
            url=url, max_depth=depth, extractor=lambda x: Soup(x, "html.parser").text
        )
        docs = loader.load()

        #TODO: support appropriate chunk size customization
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        return text_splitter.split_documents(docs)

    def load_embedding_model(service="huggingface", **kwargs):
        if service == "openai":
            #TODO: api_key etc.
            return OpenAIEmbeddings()
        else:
            return HuggingFaceEmbeddings(model_name=kwargs.get("model_name", "intfloat/multilingual-e5-large"))

    def load_chat_model(service="openai", name="gpt-4-0125-preview", **kwargs):
        #TODO: llamacpp
        #TODO: check apikey/ if api_key in kwargs:
        return ChatOpenAI(model_name=name, temperature=kwargs.get("temp", 0.2), openai_api_key=kwargs.get("api_key"))

    def create_retriever(label, embeddings, new_index=True, **kwargs):
        path = f"./{label}"
        vectorstore = None
        if new_index == False:
            vectorstore = FAISS.load_local(folder_path=path, embeddings=embeddings, allow_dangerous_deserialization=True)
        else:
            vectorstore = FAISS.from_documents(kwargs.get("docs"), embeddings)
            vectorstore.save_local(path)

        return vectorstore.as_retriever(k=2)

    def create_document_chain(llm):
        #TODO: support template customization
        SYSTEM_TEMPLATE = """
        以下の文脈に基づいて、ユーザーの質問に答えましょう。回答にはmarkdownではなくorgフォーマットを使ってください。
        文脈に質問に関連する情報がない場合は、でっち上げずに「わかりません」と答えましょう：

        <context>
        {context}
        </context>
        """
        question_answering_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    SYSTEM_TEMPLATE,
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        return create_stuff_documents_chain(llm, question_answering_prompt)

    def create_query_transform_prompt(for_history):
        #TODO: support template customization
        query = "上記の質問に関連する情報を得るために検索クエリを生成してください。そのクエリのみに反応し、それ以外には反応しないこと。"
        if for_history:
            query = "上記の会話を想定し、会話に関連する情報を得るために検索クエリを生成してください。そのクエリのみに反応し、それ以外には反応しないこと。"
        return ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder(variable_name="messages"),
                (
                    "user",
                    query
                ),
            ]
        )

    def create_conversational_retriever_chain(llm, retriever):
        document_chain = RAG.create_document_chain(llm)

        ###
        query_transform_prompt = RAG.create_query_transform_prompt(True)
        query_translate_prompt = RAG.create_query_transform_prompt(False)

        query_transforming_retriever_chain = RunnableBranch(
            (
                lambda x: len(x.get("messages", [])) == 1,
                # query_translate_prompt | llm | StrOutputParser() | retriever,
                (lambda x: x["messages"][-1].content) | retriever,
            ),
            query_transform_prompt | llm | StrOutputParser() | retriever,
        ).with_config(run_name="chat_retriever_chain")

        ###
        return RunnablePassthrough.assign(
            context=query_transforming_retriever_chain,
        ).assign(
            answer=document_chain,
        )

class Server:
    def __init__(self):
        self.setup_epc()

    def setup_epc(self, address='localhost', port=0, logfilename='rag-shell-server.log'):
        self.server = EPCServer((address, port), log_traceback=True)
        self.server.logger.setLevel(logging.DEBUG)

        ch = logging.FileHandler(filename=logfilename, mode='w')
        ch.setLevel(logging.DEBUG)
        self.server.logger.addHandler(ch)

        def setup_embedding_model(service, name):
            self.embedding = RAG.load_embedding_model(service, model_name=name)

        def setup_chat_model(service, name, temp, key):
            self.chat = RAG.load_chat_model(service, name, temp=temp, api_key=key);

        def setup_rag_chain(label, url):
            retriever = None
            if os.path.exists(f"./{label}/index.faiss"):
                retriever = RAG.create_retriever(label, self.embedding, new_index=False)
            else:
                all_splits = RAG.load_web_docs(url, 2)
                # retriever
                retriever = RAG.create_retriever(label, self.embedding, new_index=True, docs=all_splits)
                # chain

            # https://python.langchain.com/docs/use_cases/chatbots/retrieval/
            self.chain = RAG.create_conversational_retriever_chain(self.chat, retriever)

        def chat(question):
            ## TODO: history management but reduce token cost. Summarize?
            ## https://python.langchain.com/docs/use_cases/chatbots/memory_management/
            response = self.chain.invoke({"messages":[HumanMessage(content=question)]})

            response_result = response["answer"]
            for doc in response["context"]:
                response_result += f"\n#+begin_quote\n{doc.page_content}\n#+end_quote\n\n[[{doc.metadata.get("title","-")}][{doc.metadata.get("source","")}]]\n"

            return response_result
        def destroy():
            self.server.shutdown()

        self.server.register_function(setup_embedding_model)
        self.server.register_function(setup_chat_model)
        self.server.register_function(setup_rag_chain)
        self.server.register_function(chat)
        self.server.register_function(destroy)

    def main(self):
        self.server.print_port()  # needed for Emacs client
        self.server.serve_forever()


if __name__ == '__main__':
    #TODO: argument to control server address or port
    server = Server()
    server.main()
    server.logger.info('exit')
