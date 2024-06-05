import streamlit as st
import cohere
import re
import pandas as pd
import calplot
import matplotlib.pyplot as plt

api_key = st.secrets["API_KEY"]
cohere_client = cohere.Client(api_key)

def extract_event_details(description):
    prompt = (
        f"Extract the event, year, month, day, and time from this event description: \"{description}\" "
        "If the year is not specified, use 2024. Keep the year, month, and day all in number form. For example, if the month is December, the month should be 12. If the date is the 3rd, it should be 3. If the time is not specified, interpolate the time based off of the event. Provide the output in the following format:\n"
        "Event: [event]\nYear: [year]\nMonth: [month]\nDay: [day]\nTime: [time]"
    )
    
    response = cohere_client.generate(
        model='command-xlarge-nightly',
        prompt=prompt,
        max_tokens=100,
        temperature=0.7
    )
    
    extracted_text = response.generations[0].text.strip() if hasattr(response, 'generations') else "No text found"

    event_pattern = r'Event:\s*(.+)'
    year_pattern = r'Year:\s*(\d{4})'
    month_pattern = r'Month:\s*(\d{1,2})'
    day_pattern = r'Day:\s*(\d{1,2})'
    time_pattern = r'Time:\s*(.+)'
    
    event_match = re.search(event_pattern, extracted_text)
    year_match = re.search(year_pattern, extracted_text)
    month_match = re.search(month_pattern, extracted_text)
    day_match = re.search(day_pattern, extracted_text)
    time_match = re.search(time_pattern, extracted_text)
    
    event = event_match.group(1).strip() if event_match else "Event not found"
    year = int(year_match.group(1).strip()) if year_match else 2024
    month = int(month_match.group(1).strip()) if month_match else None
    day = int(day_match.group(1).strip()) if day_match else None
    time = time_match.group(1).strip() if time_match else "Time not found"
    
    return event, year, month, day, time

def is_valid_date(year, month, day):
    try:
        pd.to_datetime({'year': [year], 'month': [month], 'day': [day]})
        return True
    except (ValueError, pd.errors.OutOfBoundsDatetime):
        return False

st.title("Calendar LLM App")
st.write("This is a calendar application powered by Cohere and Streamlit.")
event_description = st.text_input("Enter a description for your event:")

if 'events' not in st.session_state:
    st.session_state.events = []

if st.button("Add Event"):
    if event_description:
        event, year, month, day, time = extract_event_details(event_description)
        
        if is_valid_date(year, month, day):
            date = pd.to_datetime({'year': [year], 'month': [month], 'day': [day]})[0]
            st.session_state.events.append({
                "Event": event,
                "Date": date,
                "Time": time
            })
            
            st.write("Event Details:")
            st.write(f"Event: {event}")
            st.write(f"Date: {date}")
            st.write(f"Time: {time}")
        else:
            st.write("Invalid date format. Please enter a valid date.")
    else:
        st.write("Please enter an event description.")

if st.session_state.events:
    events_df = pd.DataFrame(st.session_state.events)
    events_df['Count'] = 1
    events_df.set_index('Date', inplace=True)
    
    pivot_table = events_df.pivot_table(values='Count', index=events_df.index, aggfunc='sum')
    pivot_table = pivot_table.asfreq('D', fill_value=0)  # Ensure all days are present
    pivot_table = pivot_table.fillna(0)  # Replace any NaNs with 0

    st.write(pivot_table)  # Display the pivot table for debugging
    
    fig, ax = plt.subplots(figsize=(16, 8))
    calplot.yearplot(pivot_table['Count'], cmap='YlGn', ax=ax)
    plt.colorbar(ax.collections[0], ax=ax, orientation='horizontal')
    st.pyplot(fig)