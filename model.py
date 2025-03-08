import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, GlobalAveragePooling1D
from tensorflow.keras.models import Model
import numpy as np
import sqlite3  # Database connection
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 🔹 Setup SQLite Database
conn = sqlite3.connect("college_database.db")
cursor = conn.cursor()

# 🔹 Training Data (Queries and Responses)
queries = [
    "Who is the principal?",
    "Who is the correspondent?",
    "Where is the college?",
    "When was the college established?",
    "What is the college infrastructure?",
    "Who is the RRC incharge?",
    "How many clubs are there?",
    "What facilities are available in this college?",
    "Who is the HOD of the Zoology department?",
    "What are the achievements of the Zoology department?",
    "Who are the alumni of the Computer Science department?",
    "What is the vision of the Visual Communication department?",
    "List the faculty of the Zoology department.",
    "What is the syllabus for Cloud Computing?",
    "What are the unit headings in Cloud Computing?",
    "Give me the staff details of the Computer Science department.",
    "Show the student details of the Computer Science department.",
    "What is today's timetable?",
    "What is the timetable for Monday?",
    "Show the Maths department timetable for Monday.",
    "How many subjects are in the first year MSc Computer Science?",
]

syllabus_texts = [
    "Cloud Computing syllabus includes networking, virtualization, security, etc.",
    "Units in Cloud Computing: Introduction, Virtualization, Cloud Security, etc.",
]

college_texts = [
    "The principal is Dr. XYZ.",
    "The correspondent is Mr. ABC.",
    "The college is located in Tamil Nadu.",
    "The college was established in 1985.",
    "College infrastructure includes modern classrooms, laboratories, and libraries.",
    "The RRC incharge is Dr. PQR.",
    "The college has 10 different student clubs.",
]

department_texts = [
    "Facilities include computer labs, library, sports complex, and research centers.",
    "The HOD of the Zoology department is Dr. LMN.",
    "Zoology department achievements include international research collaborations.",
    "Computer Science department alumni include graduates working at Google, Microsoft, etc.",
    "The vision of the Visual Communication department is to innovate in digital media.",
    "The faculty of the Zoology department includes 15 professors.",
]

db_texts = [
    "Staff details: Dr. ABC, Dr. XYZ, Prof. PQR, etc.",
    "Student details include name, roll number, and department.",
    "Today's timetable: CS101, MA102, PH103.",
    "Monday's timetable: MA201, CS202, PH203.",
    "Maths department timetable for Monday: Algebra, Calculus, Statistics.",
    "First-year MSc Computer Science has 5 subjects.",
]

# 🔹 Labels (Categories for Classification)
labels = (
    ["College"] * len(college_texts) +
    ["Syllabus"] * len(syllabus_texts) +
    ["Department"] * len(department_texts) +
    ["DB"] * len(db_texts)
)

# 🔹 Combine all training texts
all_texts = college_texts + syllabus_texts + department_texts + db_texts

# 🔹 Tokenization and Sequence Preparation
tokenizer = Tokenizer()
tokenizer.fit_on_texts(queries + all_texts)
vocab_size = len(tokenizer.word_index) + 1

# Convert text data to numerical sequences
all_sequences = pad_sequences(tokenizer.texts_to_sequences(all_texts), maxlen=12)

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# 🔹 Define Neural Network Model
input_text = Input(shape=(12,))
embedding_layer = Embedding(input_dim=vocab_size, output_dim=16, input_length=12)(input_text)
vector = GlobalAveragePooling1D()(embedding_layer)
dense = Dense(16, activation="relu")(vector)
output = Dense(4, activation="softmax")(dense)  # 4 categories: College, Syllabus, Department, DB
print("X_train shape:", all_sequences.shape)
print("y_train shape:", encoded_labels.shape)
print("Unique labels in y_train:", np.unique(encoded_labels))

model = Model(inputs=input_text, outputs=output)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# 🔹 Train the Model
model.fit(all_sequences, np.array(encoded_labels), epochs=10, batch_size=2)

# 🔹 Prediction Function
def predict_relevance(user_input, college_details, department_details, syllabus, db_details):
    input_seq = pad_sequences(tokenizer.texts_to_sequences([user_input]), maxlen=12)

    prediction = model.predict(input_seq)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]

    # Select appropriate response
    if predicted_label == "College":
        return college_details
    elif predicted_label == "Department":
        return department_details
    elif predicted_label == "Syllabus":
        return syllabus
    elif predicted_label == "DB":
        return db_details
    else:
        return "I'm not sure how to answer that."

# 🔹 Example Usage
college_info = '''===Sports===
A 16-station multi-gym facility has been developed. In addition to this, the campus includes four volleyball courts, two basketball courts, two tennis courts, one badminton court, two cricket grounds, two football fields, three kabaddi courts, one tenni-koi court, two kho-kho courts, as well as facilities for indoor games and yoga.
'''

department_info = '''
**Present HOD of Visual Communication:** N. Vijayakumar, M.A, MSW, M.Phil. (15 yrs)
Former HOD
**Previous HODs of Visual Communication and Experience:** 
Mr. Shanmugavelayutham, 
Dr. B. Sundaresan,
Mr. B. Venkatesh
'''

syllabus_info = '''
Courses offered in PG Computer Science include Cloud Computing, Digital Image Processing, Web Services Using Laravel, .Net Programming, Deep Learning, and more.
'''

db_info = '''
Internal marks of Abirami: [(3.33, 3.0, None, 5.0, 5.0, 26.0, 15.0, None), (None, 3.0, None, None, 5.0, None, 15.0, None)]
'''

# Example Query
user_query = "mark details"
predicted_response = predict_relevance(user_query, college_info, department_info, syllabus_info, db_info)
print(f"Response: {predicted_response}")

# 🔹 Close Database Connection
conn.close()
