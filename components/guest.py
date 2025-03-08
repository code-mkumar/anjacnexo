import streamlit as st
import genai.gemini
import genai.lama
import ml.sentiment_feedback
import operation
# import operation.dboperation
# import operation.fileoperations
import operation.dboperation
import operation.fileoperations
import operation.mailoperation
import operation.otheroperation
import operation.preprocessing
import json
import genai
import operation
import os
from datetime import datetime
import random
import ml
import operation.speech

# Function to display a dynamic welcome message

    
def guest_page():
    # Initialize session state variables
    if 'qa_list' not in st.session_state:
        st.session_state.qa_list = []
    if 'username' not in st.session_state:
        st.session_state.username = ''
    if 'input' not in st.session_state:
        st.session_state.input = ""
    if 'stored_value' not in st.session_state:
        st.session_state.stored_value = ""

    # Sidebar for navigation and displaying past Q&A
    if "feedback" not in st.session_state:
        st.session_state.feedback = 0

    with st.sidebar:
        if st.button("Go to Login"):
            st.session_state.page = "login"
            st.rerun()
        
        if "qa_list" in st.session_state and len(st.session_state.qa_list) > 3 and st.session_state.feedback == 0 and len(st.session_state.qa_list) :
            # qa_list exists and its length is a multiple of 3
            
            with st.popover("feedback"):
                
                user_id = st.text_input("Email ID")
                name = st.text_input("Your Name",value=st.session_state.username,disabled=True)
                message = st.text_area("Your Feedback")
                
                if st.button("Submit Feedback"):
                    if user_id and name and message:
                        operation.dboperation.add_feedback(user_id, name, message)
                        st.balloons()
                        st.session_state.feedback=1
                        operation.mailoperation.email(name,user_id)
                    else:
                        st.warning("Please fill all the fields to submit your feedback.")
                    st.rerun()
        st.title("Chat History")
        for qa in reversed(st.session_state.qa_list):
            st.write(f"**Question:** {qa['question']}")
            st.write(f"**Answer:** {qa['answer']}")
            st.write("---")
    
            
    
    # Load text files for college and department history
    # with open("collegehistory.txt", "r") as f:
    #     collegehistory = f.read()
    # with open("departmenthistory.txt", "r") as f:
    #     departmenthistory = f.read()
    # default,default_sql=operation.fileoperations.read_default_files()
    # Display guest welcome message
    # print(feedback)
    st.title("Welcome to ANJAC AI!")
    st.write("You can explore the site as a guest, but you'll need to log in for full role-based access.")
    st.subheader(operation.otheroperation.get_dynamic_greeting())
    st.write("---")
    st.write(f"🎓 **Fun Fact:** {operation.otheroperation.get_fun_fact()}")
    # Ask for the user's name
    if not st.session_state.username:
        st.session_state.username =''
    name = ''
    if not name and not st.session_state.username:
        name=st.text_input('Enter your name:', placeholder='John', key='name')
        st.session_state.username = name
        if name :
            st.rerun()
       
    
        # st.write(f"Hello, {name}!")

    # Process questions if the username is set
    if st.session_state.username:
        role_prompt = operation.fileoperations.read_from_file('default.txt')
        folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../files/"))

        # Define the list of files to exclude
        excluded_files = {'staff_role.txt', 'staff_sql.txt', 'student_role.txt', 'student_sql.txt', 'default.txt','syllabus.txt','staff_sql(2).txt','student_sql(2).txt','data.txt'}

        # Get all files in the folder except the excluded ones
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f not in excluded_files]
        info=''
        # Print the filtered files
        text={}
        for file in files:
            text[file]=str(operation.fileoperations.read_from_file(file))

        
        for file,val in text.items():
            text[file]= str(val).replace("']['", "").strip()
            text[file]= str(val).replace("['", "").strip()
            text[file]= str(val).replace("']", "").strip()

        # st.write(text)
        chunks={}
        for file,val in text.items():
            if file=='college_history.txt':
                val ="".join(val)
                val = val.replace("']['", "").strip()
                val = val.replace("['", "").strip()
                chunks[file]=operation.preprocessing.chunk_text_by_special_character(val)
            else:
                chunks[file]=val
        # st.write(chunks)
        testlist=[]
        

  

        for file, val in chunks.items():
            if file == 'college_history.txt':
                # st.write(val)
                testlist = [entry[:50] for entry in val]  # Extract first 50 characters as a string

        
              
                
        st.write(f"Hello, {st.session_state.username}!")
        # chunks = operation.preprocessing.chunk_text(f"{collegehistory}\n{departmenthistory}")
        # question = st.chat_input("Ask your question")
        if st.button("🎤 Speak your question"):
            spoken_question = operation.speech.recognize_speech()
            st.text(f"You said: {spoken_question}")
        else:
            spoken_question = ""

        # Text Input
        # question = st.chat_input("Ask your question") or spoken_question
        if question := st.chat_input("Ask your question") or spoken_question:
            #st.write("**ANJAC AI can make mistakes. Check important info.**")
            st.markdown("**:red[ANJAC AI can make mistakes. Check important info.]**")

        if question:
            ml.sentiment_feedback.predict_and_store(question,st.session_state.qa_list)
            # Retrieve relevant chunks
            st.chat_message('user').markdown(question)
            k = 3  
            keywords = ['more', 'most', 'details', 'full information']
            if any(keyword in question.lower() for keyword in keywords):
                k = 3 
            previous = ",".join(qa_item["question"] for qa_item in st.session_state.qa_list)
            # relevant_chunks = operation.preprocessing.get_relevant_chunks(question, chunks["college_history.txt"],k,previous)
            relevant_chunks = operation.preprocessing.get_relevant_chunks(question, testlist,k,previous,chunks["college_history.txt"])
            # relevant_departments=operation.preprocessing.get_relevant_chunks(question,list(chunks.keys()))
            # st.write(relevant_departments)
            # st.write(chunks.keys())
            rel_departments=operation.preprocessing.relevent_department(question,list(chunks.keys()))
            # st.write(rel_departments)
            # st.write(relevant_chunks)
            department_chunks=''
            if rel_departments:
                for department in rel_departments:
                    department_chunks+=str(operation.fileoperations.read_from_file(department))
            # st.write(department_chunks)
            #r_chunks=relevant_chunks.append(department_chunks)
            # college_details="\n".join(department_chunks)
            relevant_chunks_with_department="\n".join(relevant_chunks)+"\n"+department_chunks
            relevant_chunks_with_department=relevant_chunks_with_department.replace("\n","")
            #st.write(relevant_chunks_with_department)
            # rel_chunk_size = len("\n\n".join(relevant_chunks))  # Size in characters
            # new_chunk=("".join(department_chunks,relevant_chunks))
            # st.write("Size:", rel_chunk_size)
            rel_chunk_size=len(relevant_chunks_with_department)
            
            chunk_Size=500

            # While the size exceeds the threshold, keep splitting and processing
            while rel_chunk_size > 2000:
                # Split the current relevant chunks into smaller parts
                rechunk = operation.preprocessing.chunk_text(relevant_chunks_with_department,chunk_Size)
                # st.info("Rechunked text:")
                # st.info(rechunk)
                
                # Get the new relevant chunks after rechunking
                relevant_chunks = operation.preprocessing.get_relevant_chunks_re(question, rechunk)
                
                # Display the length of the new relevant chunks
                # st.write("New relevant chunk size:", len(" ".join(relevant_chunks)))
                # st.info("New relevant chunks:")
                # st.info(" ".join(relevant_chunks))
                
                # Update the chunk size for the next iteration
                rel_chunk_size = len("\n\n".join(relevant_chunks))  # Size in characters
                chunk_Size-=100

            if relevant_chunks:
                relevant_chunks.append(department_chunks)
                relevant_chunks.append("college name:'AYYA NADAR JANAKI AMMAL COLLEGE' college loaction:'srivilliputur road ,sivakasi'")
            # relevant_chunks.append()

            context = "\n\n".join(relevant_chunks)

            # Display relevant context
            # st.write("Relevant context:")
            # st.write(context)

            # Query LM Studio for the answer
            
                
            result_text = genai.lama.query_lm_studio(question,context)          
            # Store the question and answer in session state
            st.session_state.qa_list.append({'question': question, 'answer': result_text})

            st.rerun()
        
        if len(st.session_state.qa_list):
            last_qa = st.session_state.qa_list[-1]  # Get the last Q&A pair
            st.chat_message("user").markdown(last_qa["question"])
            bot_response = st.chat_message("assistant").markdown(last_qa["answer"])

            
            operation.speech.speak_text(last_qa["answer"])  # Plays the answer as audio
            sentiment_mapping = [":material/thumb_down:", ":material/thumb_up:"]
            selected = st.feedback("thumbs")
        else:
            st.header("How can I help you today?")


                