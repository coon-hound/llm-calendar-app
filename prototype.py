import streamlit as st
import cohere
import re
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.shared import JsCode

# Initialize cohere client
api_key = st.secrets["API_KEY"]
cohere_client = cohere.Client(api_key)

def extract_event_details(description):
    prompt = (
        f"Extract the event, date, and time from this event description: \"{description}\" "
        "and provide the output in the following format:\n"
        "Event: [event]\nDate: [date]\nTime: [time]"
    )
    
    response = cohere_client.chat(
        message=prompt,
        model='command-xlarge-nightly',  # Ensure this matches your deployment
    )
    
    # Extracted text from Cohere
    extracted_text = response.text.strip()
    
    # Use regex to find event, date, and time
    event_pattern = r'Event:\s*(.+)'
    date_pattern = r'Date:\s*(.+)'
    time_pattern = r'Time:\s*(.+)'
    
    event_match = re.search(event_pattern, extracted_text)
    date_match = re.search(date_pattern, extracted_text)
    time_match = re.search(time_pattern, extracted_text)
    
    event = event_match.group(1).strip() if event_match else "Event not found"
    date = date_match.group(1).strip() if date_match else "Date not found"
    time = time_match.group(1).strip() if time_match else "Time not found"
    
    return event, date, time

# Title of the app
st.title("Calendar LLM App")

# Instructions
st.write("This is a calendar application powered by Cohere and Streamlit.")

# Text input for user to enter event description
event_description = st.text_input("Enter a description for your event:")

# Initialize session state for events
if 'events' not in st.session_state:
    st.session_state.events = []

# Button to process the input
if st.button("Add Event"):
    if event_description:
        event, date, time = extract_event_details(event_description)
        
        # Add the event to the session state
        st.session_state.events.append({
            "Event": event,
            "Date": date,
            "Time": time
        })
        
        # Display the extracted event details
        st.write("Event Details:")
        st.write(f"Event: {event}")
        st.write(f"Date: {date}")
        st.write(f"Time: {time}")
    else:
        st.write("Please enter an event description.")

# Convert the list of events to a DataFrame
events_df = pd.DataFrame(st.session_state.events)

# Display the events in a table using AgGrid
st.write("Your Events:")

if not events_df.empty:
    gb = GridOptionsBuilder.from_dataframe(events_df)
    gb.configure_pagination(paginationAutoPageSize=True)  # Add pagination
    gb.configure_side_bar()  # Add a sidebar
    gridOptions = gb.build()
    AgGrid(
        events_df,
        gridOptions=gridOptions,
        enable_enterprise_modules=True,
        theme='streamlit',
        update_mode='MODEL_CHANGED'
    )
else:
    st.write("No events added yet.")