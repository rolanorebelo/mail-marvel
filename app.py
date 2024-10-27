import os
import pickle
import dotenv
from google.oauth2.credentials import Credentials
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
from pymongo import MongoClient
import pickle
from datetime import datetime, timedelta
import plotly.express as px 
import calendar

# If modifying these scopes, delete the file token.pickle.
# Update SCOPES to include Calendar API
SCOPES = [
    'https://www.googleapis.com/auth/gmail.readonly',
    'https://www.googleapis.com/auth/gmail.send',
    'https://www.googleapis.com/auth/gmail.modify',
    'https://www.googleapis.com/auth/calendar',
    'https://www.googleapis.com/auth/calendar.events'
]

# Load environment variables from .env file
dotenv.load_dotenv()
uri = os.getenv("MONGODB_URI")
client = MongoClient(uri)
mydb = client["hackathon"]
mycol = mydb["filtered_emails"]

def create_calendar_view(classified_emails):
    """
    Create a calendar view with today's date highlighted and events/tasks displayed for the selected date.
    """
    try:
        # Get current date information
        today = datetime.now()
        
        # Add month selection
        months = list(calendar.month_name)[1:]
        current_month_idx = today.month - 1
        selected_month = st.selectbox("Select Month", months, index=current_month_idx)
        selected_month_num = months.index(selected_month) + 1
        
        current_year = today.year
        selected_year = st.selectbox("Select Year", range(current_year - 1, current_year + 2), index=1)
        
        # Create calendar object
        cal = calendar.monthcalendar(selected_year, selected_month_num)
        
        # Initialize session state for selected date if not exists
        if 'selected_date' not in st.session_state:
            st.session_state.selected_date = None
        
        # Collect events and tasks based on the selected date
        selected_date_str = st.session_state.selected_date or today.strftime('%Y-%m-%d')
        selected_date = datetime.strptime(selected_date_str, '%Y-%m-%d').strftime('%m-%d-%Y')
        
        todays_items = []
        
        for email, classification in classified_emails:
            if classification.Is_Spam:
                continue
            
            # Handle events
            try:
                if classification.Event_Date == selected_date:
                    todays_items.append({
                        'type': 'Event',
                        'title': classification.Event_Name,
                        'time': classification.Event_Time,
                        'Date': classification.Event_Date,
                        'location': classification.Event_Location,
                        'summary': classification.Summary
                    })
            except Exception:
                continue
                    
            # Handle tasks
            try:
                if classification.Deadline == selected_date:
                    todays_items.append({
                        'type': 'Task',
                        'title': email['subject'],
                        'steps': classification.Steps,
                        'summary': classification.Summary,
                        'Action Link': classification.Action_Link
                    })
            except Exception:
                continue
        
        # Display calendar
        st.subheader("Calendar View")
        
        # Create calendar grid with clickable dates
        cols_header = st.columns(7)
        
        # Display day names
        for i, day in enumerate(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']):
            cols_header[i].markdown(f"<div style='text-align: center'><b>{day}</b></div>", unsafe_allow_html=True)
        
        # Display calendar dates
        for week in cal:
            cols = st.columns(7)
            for i, day in enumerate(week):
                if day != 0:
                    date_str = f"{selected_year}-{selected_month_num:02d}-{day:02d}"
                    print('dateeee',date_str)
                    # Create button-like style for today's date
                    if date_str == today.strftime('%Y-%m-%d'):
                        button_style = "background-color: #0066cc; color: white; font-weight: bold; border-radius: 5px; padding: 5px;"
                        button_text = f"<div style='{button_style} text-align: center; width: 100%; cursor: pointer;'>{day}</div>"
                        with cols[i]:
                            if st.markdown(button_text, unsafe_allow_html=True):
                                st.session_state.selected_date = date_str
                    else:
                        # Normal button for other dates
                        with cols[i]:
                            if st.button(str(day), key=f"date_{date_str}", use_container_width=True):
                                st.session_state.selected_date = date_str
        
        # Display events and tasks for the selected date
        if st.session_state.selected_date:
            st.markdown("---")
            st.subheader(f"Events and Tasks for {st.session_state.selected_date}")

            if not todays_items:
                st.info("No events or tasks scheduled for this date.")
            else:
                events = [item for item in todays_items if item['type'] == 'Event']
                tasks = [item for item in todays_items if item['type'] == 'Task']

                # Display events
                if events:
                    st.markdown("### Events")
                    for event in events:
                        with st.expander(event['title'], expanded=True):
                            st.write(f"Time: {event['time']}")
                            st.write(f"Date: {event['Date']}")
                            st.write(f"Location: {event['location']}")
                            st.write(f"Summary: {event['summary']}")

                # Display tasks
                if tasks:
                    st.markdown("### Tasks")
                    for task in tasks:
                        with st.expander(task['title'], expanded=True):
                            st.write(f"Steps: {task['steps']}")
                            st.write(f"Summary: {task['summary']}")
                
    except Exception as e:
        st.error(f"Error creating calendar view: {str(e)}")


# Set OpenAI API Key
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key
else:
    raise ValueError("OpenAI API key not found in environment variables.")

# Define the email classification structure
class EmailClassification(BaseModel):
    Is_Spam: bool = Field(description="True if the email is classified as spam, False otherwise")
    Category: str = Field(description="One of the categories: Event, Pending Actions, Job Search")
    Summary: str = Field(description="A brief summary of around 100 words from the email")
    Received_Date: str = Field(description="Date and time of the email received in MM-DD-YYYY format; otherwise, empty")
    Event_Name: str = Field(description="Name of the event")
    Event_Date: str = Field(description="Date of the occurrence of the event in MM-DD-YYYY format.")
    Event_Time: str = Field(description="Time of the occurrence of the event.")
    Event_Location: str = Field(description="Venue of the event.")
    Steps: str = Field(description="Steps to follow if the category is 'Pending Actions'; otherwise, empty")
    Deadline: str = Field(description="Deadline Date for 'Pending Actions' in in MM-DD-YYYY format; otherwise, empty")
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
    
    # Load credentials if they exist.
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    
    # If there are no valid credentials, go through the OAuth flow.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Save the credentials for future use.
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    
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

def fetch_emails(service, query='', max_results=100):
    """
    Fetch emails from Gmail with optional query and limit
    
    Args:
        service: Gmail API service instance
        query: Search query string (optional)
        max_results: Maximum number of emails to fetch (default: 100)
        
    Returns:
        List of dictionaries containing email data
    """
    try:
        results = service.users().messages().list(
            userId='me', 
            q=query, 
            maxResults=max_results
        ).execute()
        
        messages = results.get('messages', [])

        email_data = []
        for message in messages:
            msg = service.users().messages().get(userId='me', id=message['id']).execute()
            
            # Extract email data
            headers = msg['payload']['headers']
            email_data.append({
                'id': msg['id'],
                'snippet': msg['snippet'],
                'from': next((header['value'] for header in headers if header['name'].lower() == 'from'), 'Unknown'),
                'subject': next((header['value'] for header in headers if header['name'].lower() == 'subject'), 'No Subject'),
                'date': next((header['value'] for header in headers if header['name'].lower() == 'date'), 'Unknown'),
                'body': get_message_body(service, message['id'])
            })
            
        return email_data
        
    except Exception as e:
        raise Exception(f"Error fetching emails: {str(e)}")

def classify_email(email):
    query = f"""
    Please classify the following email. Provide a comprehensive response in the format specified below:
    From: {email.get('from', 'Unknown')}
    Subject: {email.get('subject', 'No Subject').replace(' ', '') if email.get('subject') else 'No Subject'}
    Email Date: {email.get('date', 'Unknown')}
    Email Body: {email.get('body', '').replace(' ', '') if email.get('body') else 'No Body'}

    Classify this email as follows:
    1. If it contains event_date, event_time, and event_location, categorize it as "Event".
    2. If it contains action_link and steps, categorize it as "Pending Actions".
    3. If it's a spam, promotion, or useless email, set Is_Spam to True and leave other fields empty.
    4. If it doesn't fit into the above categories, classify it as "Other".
    Take event dates from the content of the email.
    """

    try:
        response = chain.invoke({"query": query})
        return response
    except Exception as e:
        st.error(f"An error occurred during classification: {e}")
        return None

def show_email_list(category, classified_emails):
    filtered_emails = [
        (email, classification) for email, classification in classified_emails 
        if not classification.Is_Spam and (category == "All" or classification.Category == category)
    ]
    
    st.subheader(f"Emails in Category: {category}")
    
    # Group emails by date for events and pending actions
    grouped_emails = {
        "Event": {},
        "Pending Actions": {},
        "Other": {}
    }
    
    for email, classification in filtered_emails:
        category = classification.Category
        
        if category == "Event":
            date = parse_date(classification.Event_Date)
            if date not in grouped_emails[category]:
                grouped_emails[category][date] = []
            grouped_emails[category][date].append((email, classification))
        elif category == "Pending Actions":
            deadline = parse_date(classification.Deadline)
            if deadline not in grouped_emails[category]:
                grouped_emails[category][deadline] = []
            grouped_emails[category][deadline].append((email, classification))
        else:
            date = parse_date(email['date'])
            if date not in grouped_emails["Other"]:
                grouped_emails["Other"][date] = []
            grouped_emails["Other"][date].append((email, classification))
    
    # Display grouped emails for each category
    for category, dates in grouped_emails.items():
        if dates:
            st.markdown(f"### {category}")
            for date, emails in sorted(dates.items()):
                st.markdown(f"#### {date.strftime('%Y-%m-%d')}")
                for email, classification in emails:
                    with st.expander(f"{email['subject']} ({classification.Category})"):
                        st.write(f"From: {email['from']}")
                        st.write(f"Date: {email['date']}")
                        st.write(f"Summary: {classification.Summary}")
                        if classification.Category == "Event":
                            st.write(f"Event Name: {classification.Event_Name}")
                            st.write(f"Event Date: {classification.Event_Date}")
                            st.write(f"Event Time: {classification.Event_Time}")
                            st.write(f"Event Location: {classification.Event_Location}")
                        elif classification.Category == "Pending Actions":
                            st.write(f"Steps: {classification.Steps}")
                            st.write(f"Deadline: {classification.Deadline}")
                            st.write(f"Action Link: {classification.Action_Link}")


def load_from_mongodb():
    """
    Load previously processed emails from MongoDB
    Returns a list of tuples (email, classification) in the same format as fetch_emails
    """
    try:
        # Get all documents from MongoDB
        documents = list(mycol.find({}))
        
        classified_emails = []
        for doc in documents:
            # Reconstruct email dict
            email = {
                'id': doc['email_id'],
                'from': doc['from'],
                'subject': doc['subject'],
                'date': doc['date'],
                'body': doc['body']
            }
            
            # Reconstruct classification object
            classification = EmailClassification(
                Category=doc['classification']['category'],
                Summary=doc['classification']['summary'],
                Is_Spam=doc['classification']['is_spam'],
                Event_Name=doc['classification']['event_name'],
                Event_Date=doc['classification']['event_date'],
                Event_Time=doc['classification']['event_time'],
                Event_Location=doc['classification']['event_location'],
                Steps=doc['classification']['steps'],
                Deadline=doc['classification']['deadline'],
                Action_Link=doc['classification']['action_link'],
                Received_Date=doc['date']
            )
            
            classified_emails.append((email, classification))
        
        return classified_emails
    except Exception as e:
        st.error(f"Error loading data from MongoDB: {str(e)}")
        return []

def update_email_stats(classified_emails):
    """
    Update email statistics based on classified emails
    """
    stats = {
        "totalEmails": len(classified_emails),
        "categories": {
            "Event": 0,
            "Pending Actions": 0,
            "Job Search": 0
        },
        "schedule": []
    }
    
    for email, classification in classified_emails:
        if not classification.Is_Spam:
            category = classification.Category
            if category in ["Pending Action", "Pending Reply"]:
                category = "Pending Actions"
            if category in stats["categories"]:
                stats["categories"][category] += 1
            
            # Add to schedule
            if category == "Event":
                event_date = parse_date(classification.Event_Date)
                if event_date and event_date >= datetime.now():
                    stats["schedule"].append({
                        "category": category,
                        "title": classification.Event_Name,
                        "date": event_date.strftime('%Y-%m-%d'),
                        "time": classification.Event_Time,
                        "location": classification.Event_Location
                    })
            elif category == "Pending Actions":
                deadline = parse_date(classification.Deadline)
                if deadline and deadline >= datetime.now():
                    stats["schedule"].append({
                        "category": category,
                        "title": email['subject'],
                        "date": deadline.strftime('%Y-%m-%d'),
                        "steps": classification.Steps,
                        "action_link": classification.Action_Link
                    })
    
    return stats

def parse_date(date_string):
    if not date_string:
        return None
    try:
        return parser.parse(date_string)
    except (ValueError, TypeError):
        return None

def show_email_list(category, classified_emails):
    filtered_emails = [
        (email, classification) for email, classification in classified_emails 
        if not classification.Is_Spam and (category == "All" or classification.Category == category)
    ]
    
    st.subheader(f"Emails in Category: {category}")
    
    # Group emails by date for all categories
    grouped_emails = {
        "Event": {},
        "Pending Actions": {},
        "Job Search": {}
    }
    
    for email, classification in filtered_emails:
        category = classification.Category
        if category in ["Pending Action", "Pending Reply"]:
            category = "Pending Actions"
        
        if category not in grouped_emails:
            grouped_emails[category] = {}
        
        # Get appropriate date based on category
        if category == "Event":
            date = parse_date(classification.Event_Date)
        elif category == "Pending Actions":
            date = parse_date(classification.Deadline)
        else:
            date = parse_date(email['date'])
        
        # Use a default date for None values
        if date is None:
            date = datetime.now()
        
        if date not in grouped_emails[category]:
            grouped_emails[category][date] = []
        grouped_emails[category][date].append((email, classification))
    
    # Display grouped emails for each category
    for category, dates in grouped_emails.items():
        if dates:
            st.markdown(f"### {category}")
            # Sort dates and format them properly
            sorted_dates = sorted(dates.items(), key=lambda x: x[0])
            for date, emails in sorted_dates:
                # Format date as YYYY-MM-DD without time
                formatted_date = date.strftime('%Y-%m-%d')
                st.markdown(f"#### {formatted_date}")
                
                for email, classification in emails:
                    with st.expander(f"{email['subject']} ({classification.Category})"):
                        st.write(f"From: {email['from']}")
                        st.write(f"Date: {email['date']}")
                        st.write(f"Summary: {classification.Summary}")
                        
                        if classification.Category == "Event":
                            st.write(f"Event Name: {classification.Event_Name}")
                            st.write(f"Event Time: {classification.Event_Time}")
                            st.write(f"Event Date: {classification.Event_Date}")
                            st.write(f"Event Location: {classification.Event_Location}")
                        elif classification.Category in ["Pending Action", "Pending Reply", "Pending Actions"]:
                            st.write(f"Steps: {classification.Steps}")
                            st.write(f"Deadline: {classification.Deadline}")
                            if classification.Action_Link:
                                st.write(f"Action Link: {classification.Action_Link}")

def main():
    st.set_page_config(layout="wide")
    # Halloween Theme and Animation
    st.markdown(
        """
        <style>
        /* Background color */
        body {
            background-color: #1a1a1d;
            color: #f5f5f5;
            font-family: 'Courier New', Courier, monospace;
        }

        /* Floating bats animation */
        .bat {
            position: absolute;
            animation: float 6s infinite ease-in-out;
        }

        .bat:nth-child(1) { left: 20%; top: 30%; animation-delay: 0s; }
        .bat:nth-child(2) { left: 70%; top: 10%; animation-delay: 2s; }
        .bat:nth-child(3) { left: 40%; top: 60%; animation-delay: 4s; }
        .bat:nth-child(4) { left: 80%; top: 80%; animation-delay: 1s; }

        @keyframes float {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-20px); }
        }

        /* Custom Halloween text styling */
        .header {
            text-align: center;
            color: #ff7518;
            font-size: 3em;
            margin-top: 20px;
            text-shadow: 3px 3px 5px black;
        }
        
        </style>
        
        <!-- HTML for animated bats -->
        <div class="bat">
            🦇
        </div>
        <div class="bat">
            🦇
        </div>
        <div class="bat">
            🦇
        </div>
        <div class="bat">
            🦇
        </div>

        <!-- Header for Halloween Theme -->
        <div class="header">
            🎃 Happy Halloween 2024 🎃
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.title('Mail Marvel')

    # Initialize session state variables
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

    # Authentication section
    if not st.session_state.authenticated:
        st.write("Please authenticate with your Google account to access Gmail and Calendar.")
        if st.button('Authenticate with Google'):
            try:
                creds = authenticate()
                st.session_state.service = build('gmail', 'v1', credentials=creds)
                st.session_state.calendar_service = build('calendar', 'v3', credentials=creds)
                st.session_state.authenticated = True
                st.success('Authentication successful!')

                # Load cached data from MongoDB after authentication
                cached_emails = load_from_mongodb()
                if cached_emails:
                    st.info(f"Loaded {len(cached_emails)} previously processed emails from cache.")
                    st.session_state.classified_emails = cached_emails
                    st.session_state.email_stats = update_email_stats(cached_emails)
                st.rerun()
                
            except Exception as e:
                st.error(f'Authentication failed: {str(e)}')
    else:
        # Create sidebar for navigation
        st.sidebar.title("Navigation")
        page = st.sidebar.radio("Go to", ["Email Processing", "Calendar", "Analytics"])

        if page == "Calendar":
           # st.header("Calendar View")
            create_calendar_view(st.session_state.classified_emails)

        elif page == "Email Processing":
           # st.header("Email Processing")

            # Search and fetch section
            col1, col2 = st.columns([3, 1])
            with col1:
                query = st.text_input('Enter search query (optional):', 
                                    help="Enter keywords to filter emails. Leave empty to fetch recent emails.")
            with col2:
                max_results = st.number_input('Number of emails to fetch:', 
                                            min_value=1, max_value=100, value=20)

            if st.button('Fetch and Process New Emails'):
                with st.spinner('Fetching and classifying new emails...'):
                    try:
                        # Fetch and process new emails
                        emails = fetch_emails(st.session_state.service, query=query, max_results=max_results)
                        
                        classified_emails = []
                        progress_bar = st.progress(0)
                        
                        for idx, email in enumerate(emails):
                            # Check if email already exists in MongoDB
                            existing_email = mycol.find_one({"email_id": email['id']})
                            if not existing_email:
                                classification = classify_email(email)
                                if classification and not classification.Is_Spam:
                                    classified_emails.append((email, classification))
                                    
                                    # Create MongoDB document
                                    email_doc = {
                                        "email_id": email['id'],
                                        "from": email['from'],
                                        "subject": email['subject'],
                                        "date": email['date'],
                                        "body": email['body'],
                                        "classification": {
                                            "category": classification.Category,
                                            "summary": classification.Summary,
                                            "is_spam": classification.Is_Spam,
                                            "event_name": classification.Event_Name,
                                            "event_date": classification.Event_Date,
                                            "event_time": classification.Event_Time,
                                            "event_location": classification.Event_Location,
                                            "steps": classification.Steps,
                                            "deadline": classification.Deadline,
                                            "action_link": classification.Action_Link
                                        },
                                        "processed_at": datetime.now()
                                    }
                                    
                                    # Update or insert the document in MongoDB
                                    result = mycol.update_one(
                                        {"email_id": email['id']},  # Filter to find the document by email_id
                                        {"$set": email_doc},        # Set the fields from email_doc
                                        upsert=True                 # Insert if no document matches the filter
                                    )
                                    
                                    if result.matched_count > 0:
                                        print(f"Email with ID: {email['id']} updated successfully.")
                                    elif result.upserted_id:
                                        print(f"New email with ID: {email['id']} inserted successfully.")
                            
                            progress_bar.progress((idx + 1) / len(emails))
                        
                        # Combine new and cached emails
                        if classified_emails:
                            st.session_state.classified_emails.extend(classified_emails)
                            st.session_state.email_stats = update_email_stats(st.session_state.classified_emails)
                            st.success(f"Processed {len(classified_emails)} new emails.")
                        else:
                            st.info("No new emails to process.")
                            
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")

            # Display category filter and emails
            if st.session_state.classified_emails:
                # Get unique categories from classified emails
                categories = set()
                for email, classification in st.session_state.classified_emails:
                    if not classification.Is_Spam:
                        category = classification.Category
                        if category in ["Pending Action", "Pending Reply"]:
                            category = "Pending Actions"
                        categories.add(category)
                
                # Convert to list and add "All" option
                category_list = ["All"] + sorted(list(categories))
                
                # Create the dropdown
                selected_category = st.selectbox(
                    "Filter by Category:",
                    category_list,
                    index=0
                )
                
                # Display filtered emails
                show_email_list(selected_category, st.session_state.classified_emails)

        elif page == "Analytics":
            st.header("Email Analytics")

            # Display email statistics
            st.subheader("Email Statistics")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Emails", st.session_state.email_stats["totalEmails"])
            col2.metric("Events", st.session_state.email_stats["categories"]["Event"])
            col3.metric("Pending Actions", st.session_state.email_stats["categories"]["Pending Actions"])
            col4.metric("Job Search", st.session_state.email_stats["categories"]["Job Search"])

            # Create a pie chart for email categories
            fig = go.Figure(data=[go.Pie(labels=list(st.session_state.email_stats["categories"].keys()), 
                                       values=list(st.session_state.email_stats["categories"].values()))])
            fig.update_layout(title_text="Email Categories Distribution")
            st.plotly_chart(fig)

if __name__ == "__main__":
    main()
