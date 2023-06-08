from dotenv import load_dotenv, find_dotenv
import streamlit as st
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate


def chunks_from_yt(url):
    video = YoutubeLoader.from_youtube_url(url)
    transcript = video.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(transcript)

    return texts

def notes_gen(texts, embeddings):

    search = Chroma.from_documents(texts, embeddings)

    query=""" Get all the important, helpful, useful, productive topics in this video, ignore
        topics like sponsorship, subscribing of the youtube channel and liking the video
    """

    relevant = search.similarity_search(query, k=4)

    relevant_content = " ".join([data.page_content for data in relevant])

    template = """
        You are an expert topics identifier and notes maker, who can develop notes
        on youtube videos based on the video's transcript.

        Develop topic wise notes by searching important topics that cover the entire video with
        the following transcript: {docs}
    
    """
    notes_template = PromptTemplate(
        input_variables=["docs"],
        template=template
    )


    llm = OpenAI()
    notes_chain = LLMChain(llm=llm, prompt=notes_template)

    response = notes_chain.run(docs=relevant_content)

    st.write(response)


def main():

    load_dotenv(find_dotenv())

    st.title('Youtube video to notes')

    url = st.text_input("Here goes the youtube video url")

    st.caption("*Works fine only with caption enabled videos")


    embeddings = OpenAIEmbeddings()

    if url:
        texts = chunks_from_yt(url)

        notes_gen(texts, embeddings)
        


if __name__ == "__main__":
    main()
