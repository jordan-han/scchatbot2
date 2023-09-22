import os
import openai
import streamlit as st
from streamlit_chat import message

# Setting API key
# openai.api_key = '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'
# os.environ["OPENAI_API_KEY"] = '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'
os.environ["OPENAI_API_KEY"] = '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'

PINECONE_API_KEY = '25f679ed-2e29-4df2-ae2a-d75256e8dfe7' # find at app.pinecone.io
PINECONE_API_ENV = 'northamerica-northeast1-gcp' # next to api key in console

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone

embeddings = OpenAIEmbeddings()

# initialize pinecone
pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_API_ENV
)
index = pinecone.Index(index_name='example-index-index') # connect to index

st.title('Singapore Customs experimental GPT Chatbot')

# Landing welcome
message("Welcome! How may I assist you? :)")

# Initialise session state variables
if'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'messages' not in st.session_state:
    st.session_state['messages'] = [{'role':'system', 'content':'You are a helpful assistant'}]

if 'model_name' not in st.session_state:
    st.session_state['model_name'] = []
if 'cost' not in st.session_state:
    st.session_state['cost'] = []
if 'total_tokens' not in st.session_state:
    st.session_state['total_tokens'] = []
if 'total_cost' not in st.session_state:
    st.session_state['total_cost'] = 0.0

# Sidebar - let user choose model, show total cost of current conversation, and let user clear the current conversation
st.sidebar.title("Sidebar")
model_name = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-4"))
counter_placeholder = st.sidebar.empty()
counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")
clear_button = st.sidebar.button("Clear Conversation", key="clear")

# Map model names to OpenAI model IDs
if model_name == "GPT-3.5":
    model = "gpt-3.5-turbo"
else:
    model = "gpt-4"

# reset everything
if clear_button:
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    st.session_state['number_tokens'] = []
    st.session_state['model_name'] = []
    st.session_state['cost'] = []
    st.session_state['total_cost'] = 0.0
    st.session_state['total_tokens'] = []
    counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")

response_container = st.container()

from langchain import PromptTemplate

def generate_context(user_input):
    query_embedding = openai.Embedding.create(input = user_input, model='text-embedding-ada-002')['data'][0]['embedding'] # Get the embedding of query using OpenAI
    chunks = index.query(query_embedding, top_k=2, include_metadata=True) # Get the top 3 chunks most similar to user_input
   
    global context
    context = []
    for i in range(2):
        context.append(chunks.matches[i].metadata['text'])
   
    # Prepare PromptTemplate (template instructions + context + query)
    template = """

        As a Singapore Customs officer, your goal is to provide highly accurate and helpful information to enquiries from traders. \
        You should strictly answer the queries based on only the context provided. \
        If you don't know the answer, just say that you don't know, don't try to make up an answer. \
        You must only provide information that appear in the sources and not make up one yourself.
        Be funny but professional in the way you answer.
       
        {context} \

        Question: {query} \

    Answer: """
   
    prompt_template = PromptTemplate(
        template = template,
        input_variables = ['context', 'query']    
    )
   
    prompt = prompt_template.format(
        query = user_input,
        context = context)
   
    return prompt

# generate a response 2
def generate_response(prompt):
   
    st.session_state['messages'].append({'role':'user','content':generate_context(prompt)})

    completion = openai.ChatCompletion.create(
        model = 'gpt-4',
        messages = st.session_state['messages']
    )
   
    # output from openai api
    response = completion['choices'][0]['message']['content']
   
    st.session_state['messages'].append({"role": "assistant", "content": response})
   
    total_tokens = completion.usage.total_tokens
    prompt_tokens = completion.usage.prompt_tokens
    completion_tokens = completion.usage.completion_tokens
   
    return response, total_tokens, prompt_tokens, completion_tokens

# container for chat history
response_container = st.container()
# container for text box
container = st.container()

with container: # y necessary??

    with st.form(key='my_form', clear_on_submit=True):
       
        user_input = st.text_area(label='You:', key='input', height=100)
        submit_button = st.form_submit_button('Submit')
       
        if user_input and submit_button:
           
            output, total_tokens, prompt_tokens, completion_tokens = generate_response(user_input)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)
            st.session_state['model_name'].append(model_name)
            st.session_state['total_tokens'].append(total_tokens)

            # from https://openai.com/pricing#language-models
            if model_name == "GPT-3.5":
               cost = total_tokens * 0.002 / 1000
            else:
                   cost = (prompt_tokens * 0.03 + completion_tokens * 0.06) / 1000

            st.session_state['cost'].append(cost)
            st.session_state['total_cost'] += cost
                   
           
if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))
            st.write(
                f"Model used: {st.session_state['model_name'][i]}; Number of tokens: {st.session_state['total_tokens'][i]}; Cost: ${st.session_state['cost'][i]:.5f}")
            counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")


st.write('Sources used to generate reply:')
st.write(context)