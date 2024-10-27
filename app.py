import os
import pickle
import dotenv
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import streamlit as st
import base64
import pandas as pd
import plotly.graph_objects as go
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from dateutil import parser

# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly', "https://www.googleapis.com/auth/gmail.send"]

# Load environment variables from .env file
dotenv.load_dotenv()

# Set OpenAI API Key
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key
else:
    raise ValueError("OpenAI API key not found in environment variables.")

# Define the email classification structure
class EmailClassification(BaseModel):
    Category: str = Field(description="One of the categories: Event, Pending Actions, Job Search")
    Summary: str = Field(description="A brief summary of around 100 words from the email")
    Date: str = Field(description="Date and time of the event if present; otherwise, empty")
    Event_Name: str = Field(description="Name of the event")
    Event_Date: str = Field(description="Date of the occurrence of the event.")
    Event_Time: str = Field(description="Time of the occurrence of the event.")
    Event_Location: str = Field(description="Venue of the event.")
    Steps: str = Field(description="Steps to follow if the category is 'Pending Actions'; otherwise, empty")
    Deadline: str = Field(description="Deadline Date for 'Pending Actions'; otherwise, empty")
    Action_Link: str = Field(description="Link to complete the Pending Action")

# Set up the ChatOpenAI model and parser
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
parser = PydanticOutputParser(pydantic_object=EmailClassification)

prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = prompt | model | parser

def authenticate():
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    
    return creds

def get_message_body(service, message_id):
    try:
        message = service.users().messages().get(userId='me', id=message_id, format='full').execute()
        payload = message['payload']
        parts = payload.get('parts')
        
        if parts:
            for part in parts:
                if part['mimeType'] == 'text/plain':
                    data = part['body']['data']
                    text = base64.urlsafe_b64decode(data).decode()
                    return text
        else:
            data = payload['body']['data']
            text = base64.urlsafe_b64decode(data).decode()
            return text
    except Exception as e:
        return f"An error occurred: {str(e)}"

def fetch_emails(service, query=''):
    results = service.users().messages().list(userId='me', q=query, maxResults=10).execute()
    messages = results.get('messages', [])

    email_data = []
    for message in messages:
        msg = service.users().messages().get(userId='me', id=message['id']).execute()
        email_data.append({
            'id': msg['id'],
            'snippet': msg['snippet'],
            'from': next((header['value'] for header in msg['payload']['headers'] if header['name'] == 'From'), 'Unknown'),
            'subject': next((header['value'] for header in msg['payload']['headers'] if header['name'] == 'Subject'), 'No Subject'),
            'date': next((header['value'] for header in msg['payload']['headers'] if header['name'] == 'Date'), 'Unknown'),
            'body': get_message_body(service, message['id'])
        })
    return email_data

def classify_email(email):
    query = f"""
    Please classify the following email in detail. Provide a comprehensive response in the format specified below, ensuring you address each aspect thoroughly:
    From: {email['from']}
    Subject: {email['subject']}
    Email Date: {email['date']}
    Email Body: {email['body']}
    """

    try:
        response = chain.invoke({"query": query})
        return response
    except Exception as e:
        st.error(f"An error occurred during classification: {e}")
        return None

def parse_date(date_string):
    try:
        parsed_date = parser.parse(date_string)
        return parsed_date.strftime('%m-%d-%Y')  # Format: MM-DD-YYYY
    except ValueError:
        return date_string  # Return original string if parsing fails

def show_email_list(category, classified_emails):
    filtered_emails = classified_emails if category == "All" else [
        (email, classification) for email, classification in classified_emails 
        if classification.Category == category
    ]
    
    st.subheader(f"Emails in Category: {category}")
    
    for email, classification in filtered_emails:
        with st.expander(f"{email['subject']} ({classification.Category})"):
            st.write(f"From: {email['from']}")
            st.write(f"Date: {email['date']}")
            st.write(f"Summary: {classification.Summary}")
            if classification.Category == "Event":
                st.write(f"Event Name: {classification.Event_Name}")
                st.write(f"Event Date: {classification.Event_Date}")
                st.write(f"Event Time: {classification.Event_Time}")
                st.write(f"Event Location: {classification.Event_Location}")
            elif classification.Category in ["Pending Action", "Pending Reply"]:
                st.write(f"Steps: {classification.Steps}")
                st.write(f"Deadline: {classification.Deadline}")
                st.write(f"Action Link: {classification.Action_Link}")

def main():
    st.set_page_config(layout="wide")
    st.title('Email Productivity Dashboard')

    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.email_stats = {
            "totalEmails": 0,
            "categories": {
                "Event": 0,
                "Pending Actions": 0,
                "Job Search": 0
            },
            "schedule": []
        }
        st.session_state.classified_emails = []

    if not st.session_state.authenticated:
        if st.button('Authenticate with Gmail'):
            try:
                creds = authenticate()
                st.session_state.service = build('gmail', 'v1', credentials=creds)
                st.session_state.authenticated = True
                st.success('Authentication successful!')
                st.rerun()
            except Exception as e:
                st.error(f'Authentication failed: {str(e)}')
    else:
        # Fetch and Classify Emails
        query = st.text_input('Enter search query (optional):')
        if st.button('Fetch and Classify Emails'):
            with st.spinner('Fetching and classifying emails...'):
                emails = fetch_emails(st.session_state.service, query)
                st.session_state.email_stats["totalEmails"] = len(emails)
                
                classified_emails = []
                for email in emails:
                    classification = classify_email(email)
                    if classification is None:
                        continue  # Skip this email if classification failed
                    classified_emails.append((email, classification))
                    
                    # Update email stats based on classification
                    category = classification.Category
                    if category in ["Pending Action", "Pending Reply"]:
                        category = "Pending Actions"  # Combine categories
                    if category in st.session_state.email_stats["categories"]:
                        st.session_state.email_stats["categories"][category] += 1
                    else:
                        st.warning(f"Unknown category: {category}")
                    
                    # Add to schedule
                    if category == "Event":
                        schedule_item = {
                            "category": category,
                            "title": classification.Event_Name,
                            "date": parse_date(classification.Event_Date),
                            "time": classification.Event_Time,
                            "location": classification.Event_Location
                        }
                    elif category == "Pending Actions":
                        schedule_item = {
                            "category": category,
                            "title": email['subject'],
                            "date": parse_date(classification.Deadline),
                            "steps": classification.Steps
                        }
                    st.session_state.email_stats["schedule"].append(schedule_item)
                
                st.session_state.classified_emails = classified_emails

        # Display Email Stats
        st.subheader("Email Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Emails", st.session_state.email_stats["totalEmails"])
        with col2:
            st.metric("Event Emails", st.session_state.email_stats["categories"]["Event"])
        with col3:
            st.metric("Pending Actions", st.session_state.email_stats["categories"]["Pending Actions"])

        # Schedule Visualization
        if st.session_state.email_stats["schedule"]:
            st.subheader("Upcoming Events and Deadlines")

            schedule_df = pd.DataFrame(st.session_state.email_stats["schedule"])
            schedule_df["date"] = pd.to_datetime(schedule_df["date"], errors="coerce").dt.strftime('%m-%d-%Y')  # Date as MM-DD-YYYY

            # Customize table appearance
            fig = go.Figure(data=[go.Table(
                header=dict(
                    values=list(schedule_df.columns),
                    fill_color="royalblue",
                    align="center",
                    font=dict(color="white", size=14),
                    line=dict(color="darkslategray"),
                    height=30
                ),
                cells=dict(
                    values=[schedule_df[col] for col in schedule_df.columns],
                    fill_color="lightgrey",
                    align="center",
                    font=dict(color="black", size=12),
                    line=dict(color="darkslategray"),
                    height=25
                )
            )])
            
            # Add styling to make the table stand out
            fig.update_layout(margin=dict(l=0, r=0, t=10, b=10), height=400)

            st.plotly_chart(fig)

        # Email Lists by Category
        category_filter = st.selectbox("Filter by Category", ["All", "Event", "Pending Actions", "Job Search"])
        show_email_list(category_filter, st.session_state.classified_emails)

if __name__ == "__main__":
    main()
