import os
import tempfile
import pandas as pd
import gradio as gr
import faiss
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.retrievers import EnsembleRetriever
from dotenv import load_dotenv

load_dotenv()

# ========== CONFIGURATION ==========
BETA_PROMPT = """You are Beta, the friendly AI assistant for UTM's Mentor-Mentee program. Your personality:
- Always helpful and enthusiastic 🤗
- Professional but approachable 👩💼
- Provide complete information from available records 📚
- Use simple, clear language with occasional emojis
- If unsure, offer to contact human coordinators 👥

Context: {context}
Question: {question}
Beta's answer:"""

# ========== DATA HANDLING ==========
def load_data():
    """Load and clean mentor/mentee data from Excel files"""
    try:
        # Load mentor data
        mentor_df = pd.read_excel(
            "Mentor_detail.xlsx",
            sheet_name="Sheet1",
            skiprows=2,
            usecols="A:I",
            names=["No", "Name", "Email", "Contact No.", "i-Kohza", "Expertise", "Availability", "Username", "Password"],
            dtype={"Username": str, "Password": str}
        ).dropna(subset=["Name"])
        
        # Clean and validate data
        mentor_df["Username"] = mentor_df["Username"].str.strip().str.lower()
        mentor_df["Password"] = mentor_df["Password"].str.strip()
        mentor_df = mentor_df.dropna(subset=["Username", "Password"])

    except Exception as e:
        print(f"Mentor data error: {str(e)}")
        mentor_df = pd.DataFrame()

    try:
        # Load mentee data
        xls = pd.ExcelFile("Mentee_details.xlsx")
        mentee_df = pd.read_excel(
            xls,
            sheet_name=0,
            skiprows=2,
            usecols="A:J",
            names=["No", "Name", "Metric Number", "Email", "Department", "Main Supervisor", "i-Kohza", "Title", "Username", "Password"],
            dtype={"Username": str, "Password": str}
        )
        
        # Clean and validate data
        mentee_df["Username"] = mentee_df["Username"].str.strip().str.lower()
        mentee_df["Password"] = mentee_df["Password"].str.strip()
        mentee_df = mentee_df.dropna(subset=["Username", "Password"])

    except Exception as e:
        print(f"Mentee data error: {str(e)}")
        mentee_df = pd.DataFrame()

    return mentor_df, mentee_df

def create_documents(mentor_df, mentee_df):
    """Create LangChain documents from DataFrames"""
    mentor_docs = [
        Document(
            page_content=(
                f"Mentor Profile: {row['Name']}\n"
                f"Expertise: {row['Expertise']}\n"
                f"Availability: {row['Availability']}\n"
                f"Contact: {row['Contact No.']}\n"
                f"Email: {row['Email']}\n"
                f"Research Area: {row['i-Kohza']}"
            ),
            metadata={"source": "mentor", "username": row["Username"]}
        ) for _, row in mentor_df.iterrows()
    ]

    mentee_docs = [
        Document(
            page_content=(
                f"Mentee Profile: {row['Name']}\n"
                f"Project: {row['Title']}\n"
                f"Department: {row['Department']}\n"
                f"Supervisor: {row['Main Supervisor']}\n"
                f"Research Area: {row['i-Kohza']}"
            ),
            metadata={"source": "mentee", "username": row["Username"]}
        ) for _, row in mentee_df.iterrows()
    ]
    return mentor_docs, mentee_docs

# ========== VECTOR STORES ==========
def initialize_stores():
    """Initialize all vector databases"""
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    
    # PDF storage
    pdf_index = faiss.IndexFlatL2(1536)
    global pdf_db
    pdf_db = FAISS(embeddings, pdf_index, InMemoryDocstore(), {})
    
    # Mentor/Mentee storage
    mentor_df, mentee_df = load_data()
    mentor_docs, mentee_docs = create_documents(mentor_df, mentee_df)
    
    global mentor_db, mentee_db
    mentor_db = FAISS.from_documents(mentor_docs, embeddings)
    mentee_db = FAISS.from_documents(mentee_docs, embeddings)

initialize_stores()

# ========== AUTHENTICATION ==========
def authenticate(username, password):
    """Secure authentication with case insensitivity"""
    try:
        clean_username = str(username).strip().lower()
        clean_password = str(password).strip()
        
        print(f"\n🔐 Login attempt: {clean_username}")
        
        mentor_df, mentee_df = load_data()
        
        # Check mentors
        if not mentor_df.empty:
            mentor_match = mentor_df[
                (mentor_df["Username"] == clean_username) &
                (mentor_df["Password"] == clean_password)
            ]
            if not mentor_match.empty:
                print(f"✅ Mentor authenticated: {clean_username}")
                return {"type": "mentor", **mentor_match.iloc[0].to_dict()}
        
        # Check mentees
        if not mentee_df.empty:
            mentee_match = mentee_df[
                (mentee_df["Username"] == clean_username) &
                (mentee_df["Password"] == clean_password)
            ]
            if not mentee_match.empty:
                print(f"✅ Mentee authenticated: {clean_username}")
                return {"type": "mentee", **mentee_match.iloc[0].to_dict()}
        
        print("❌ Invalid credentials")
        return None
    except Exception as e:
        print(f"Auth error: {str(e)}")
        return None

# ========== PROFILE MANAGEMENT ==========
def save_profile(original_user, new_data, role):
    """Save profile changes to Excel"""
    try:
        mentor_df, mentee_df = load_data()
        df = mentor_df if role == "mentor" else mentee_df
        
        # Validate username
        new_username = new_data["Username"].strip().lower()
        if new_username != original_user.lower():
            all_users = pd.concat([mentor_df["Username"], mentee_df["Username"]])
            if new_username in all_users.str.lower().values:
                return "🚫 Username already exists!"
        
        # Update data
        mask = df["Username"].str.lower() == original_user.lower()
        for col in df.columns:
            if col in new_data and col != "No":
                df.loc[mask, col] = new_data[col]
        
        # Save to Excel
        file_path = "Mentor_detail.xlsx" if role == "mentor" else "Mentee_details.xlsx"
        with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            df.to_excel(writer, index=False, header=False, startrow=2)
        
        return "✅ Profile updated successfully!"
    except Exception as e:
        return f"🚫 Error: {str(e)}"

# ========== PDF HANDLING ==========
def handle_pdf(file):
    """Process PDF files securely"""
    tmp_path = None
    try:
        # Validate file
        if not file.name.lower().endswith('.pdf'):
            return "⚠️ Please upload a valid PDF file"
        
        if os.path.getsize(file.name) > 50 * 1024 * 1024:
            return "⚠️ File size exceeds 50MB limit"
        
        # Create temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            with open(file.name, "rb") as uploaded_file:
                tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        
        # Process PDF
        loader = PyPDFLoader(tmp_path)
        pages = loader.load_and_split()
        pdf_db.add_documents(pages)
        
        return f"✅ Successfully processed {len(pages)} pages from '{os.path.basename(file.name)}'"
    except Exception as e:
        return f"🚫 PDF Error: {str(e)}"
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

# ========== CHAT FUNCTION ==========
def chat_with_beta(message, history, user_info):
    """Main chat function with Beta's personality"""
    try:
        role = user_info.get("type", "") if user_info else ""
        
        # Configure retrievers
        base_retriever = mentee_db.as_retriever(search_kwargs={"k": 100}) if role == "mentor" else mentor_db.as_retriever(search_kwargs={"k": 100})
        pdf_retriever = pdf_db.as_retriever(search_kwargs={"k": 50})
        
        # Combine retrievers
        ensemble_retriever = EnsembleRetriever(
            retrievers=[base_retriever, pdf_retriever],
            weights=[0.7, 0.3]
        )
        
        # Create QA chain
        qa = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2),
            chain_type="stuff",
            retriever=ensemble_retriever,
            chain_type_kwargs={
                "prompt": PromptTemplate(
                    template=BETA_PROMPT,
                    input_variables=["context", "question"]
                )
            }
        )
        
        response = qa.run(message)
        return "", history + [(message, f"🤖 Beta: {response}")]
    except Exception as e:
        return "", history + [(message, f"🤖 Beta: ⚠️ Error: {str(e)}")]

# ========== GRADIO INTERFACE ==========
with gr.Blocks(title="UTM Mentor-Mentee System", theme=gr.themes.Soft()) as app:
    user_state = gr.State()
    original_uname = gr.State("")
    
    # Login UI
    with gr.Column(visible=True, elem_id="login") as login_ui:
        gr.Markdown("# 🎓 UTM Mentor-Mentee System")
        username = gr.Textbox(label="Username")
        password = gr.Textbox(label="Password", type="password")
        login_btn = gr.Button("Login →", variant="primary")
    
    # Chat UI
    with gr.Column(visible=False, elem_id="chat") as chat_ui:
        with gr.Row():
            gr.Markdown("## 💬 Chat with Beta")
            profile_btn = gr.Button("👤 My Profile")
            logout_btn = gr.Button("🚪 Logout")
        
        chatbot = gr.Chatbot(height=500)
        msg = gr.Textbox(label="Your message...", placeholder="Ask Beta anything...")
        
        with gr.Row():
            pdf_upload = gr.UploadButton(
                "📁 Upload PDF (Max 50MB)", 
                file_types=[".pdf"],
                file_count="single"
            )
            pdf_status = gr.Textbox(label="PDF Status", interactive=False)
    
    # Profile UI
    with gr.Column(visible=False, elem_id="profile") as profile_ui:
        with gr.Row():
            gr.Markdown("## 🛠 Edit Profile")
            back_btn = gr.Button("← Back")
        
        with gr.Row():
            with gr.Column():
                name = gr.Textbox(label="Full Name")
                email = gr.Textbox(label="Email")
                uname = gr.Textbox(label="Username")
                pwd = gr.Textbox(label="Password", type="password")
            
            with gr.Column(visible=False) as mentor_col:
                expertise = gr.Textbox(label="Expertise")
                availability = gr.Dropdown(["Available", "Busy"], label="Availability")
            
            with gr.Column(visible=False) as mentee_col:
                department = gr.Textbox(label="Department")
                project = gr.Textbox(label="Project")
        
        save_btn = gr.Button("💾 Save Changes")
        status = gr.Markdown()

    # ========== EVENT HANDLERS ==========
    login_btn.click(
        authenticate,
        [username, password],
        [user_state]
    ).success(
        lambda u: (gr.update(visible=u is None), gr.update(visible=u is not None), u.get("Username", "") if u else ""),
        [user_state],
        [login_ui, chat_ui, original_uname]
    )
    
    profile_btn.click(
        lambda: (gr.update(visible=False), gr.update(visible=True)),
        outputs=[chat_ui, profile_ui]
    ).then(
        lambda u: [
            u.get("Name", ""),
            u.get("Email", ""),
            u.get("Username", ""),
            u.get("Password", ""),
            u.get("Expertise", ""),
            u.get("Availability", "Available"),
            u.get("Department", ""),
            u.get("Title", ""),
            gr.update(visible=u.get("type") == "mentor"),
            gr.update(visible=u.get("type") == "mentor"),
            gr.update(visible=u.get("type") == "mentee"),
            gr.update(visible=u.get("type") == "mentee")
        ],
        inputs=user_state,
        outputs=[name, email, uname, pwd, expertise, availability, department, project, expertise, availability, department, project]
    )
    
    save_btn.click(
        lambda: [
            save_profile(
                original_uname.value,
                {
                    "Name": name.value,
                    "Email": email.value,
                    "Username": uname.value,
                    "Password": pwd.value,
                    "Expertise": expertise.value,
                    "Availability": availability.value,
                    "Department": department.value,
                    "Title": project.value
                },
                user_state.value.get("type", "")
            ),
            gr.update(visible=False),
            gr.update(visible=True)
        ],
        outputs=[status, profile_ui, chat_ui]
    )
    
    back_btn.click(
        lambda: (gr.update(visible=True), gr.update(visible=False)),
        outputs=[chat_ui, profile_ui]
    )
    
    logout_btn.click(
        lambda: (gr.update(visible=True), gr.update(visible=False)),
        outputs=[login_ui, chat_ui]
    )
    
    pdf_upload.upload(
        handle_pdf,
        inputs=pdf_upload,
        outputs=pdf_status
    )
    
    msg.submit(
        chat_with_beta,
        [msg, chatbot, user_state],
        [msg, chatbot]
    )

if __name__ == "__main__":
    app.launch(share=True)