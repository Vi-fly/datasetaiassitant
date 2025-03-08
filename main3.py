import os
import sqlite3
import streamlit as st
from datetime import datetime, time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
import pandas as pd
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
import json
import plotly.express as px
import tempfile
import time
from jinja2 import Template
from datetime import datetime, timedelta



# ====== Unified Database Handler ======
def get_db_connection():
    """Create a new database connection for each operation"""
    return sqlite3.connect('test.db', check_same_thread=False)

def safe_db_query(query, params=None, read=True):
    """Universal database query executor"""
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(query, params or ())
        
        if read:
            columns = [desc[0] for desc in cur.description] if cur.description else []
            data = cur.fetchall()
            return columns, data
        else:
            conn.commit()
            return cur.rowcount
            
    except sqlite3.Error as e:
        st.error(f"Database error: {str(e)}")
        return None
    finally:
        if conn:
            conn.close()

# ====== Modified Task Count Example ======
def get_task_count():
    """Safe task counter with error handling"""
    try:
        columns, data = safe_db_query("SELECT COUNT(*) FROM TASKS")
        return data[0][0] if data else 0
    except Exception as e:
        st.error(f"Count error: {str(e)}")
        return 0

def get_total_tasks():
    try:
        columns, data = safe_db_query("SELECT COUNT(*) FROM TASKS")
        return data[0][0] if data else 0
    except Exception as e:
        st.error(f"Error getting task count: {str(e)}")
        return 0

def get_overdue_tasks():
    try:
        with sqlite3.connect('test.db') as conn:  # Use context manager
            result = pd.read_sql(
                "SELECT COUNT(*) FROM TASKS WHERE DEADLINE < CURRENT_TIMESTAMP", 
                conn
            )
            return result.iloc[0,0]
    except Exception as e:
        st.error(f"Error fetching overdue tasks: {str(e)}")
        return 0



conn = sqlite3.connect('test.db')
conn.execute("""
    CREATE TABLE IF NOT EXISTS RESOURCES (
        ID INTEGER PRIMARY KEY AUTOINCREMENT,
        NAME TEXT NOT NULL,
        TYPE TEXT NOT NULL,
        STATUS TEXT DEFAULT 'Available',
        ASSIGNED_TO INTEGER,
        LAST_MAINTENANCE DATE
    )
""")
conn.commit()
conn.close()


# Load environment variables
load_dotenv('.env')

# Initialize ChatGroq
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile",
)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize SQL Agent
try:
    db_agent = SQLDatabase.from_uri(
        "sqlite:///test.db",
        include_tables=['CONTACTS', 'TASKS'],
        sample_rows_in_table_info=2
    )
    
    llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile",
)
    agent_executor = create_sql_agent(
        llm=llm,
        db=db_agent,
        verbose=True,
        top_k=5,
        max_iterations=10
    )
except Exception as e:
    st.error(f"Failed to initialize SQL agent: {e}")

# Helper Functions
def generate_sql_query(prompt: str, action: str) -> str:
    """Generate SQL query based on selected action and user input."""
    system_prompts = {
        "add": (
            "You are an expert in generating SQL INSERT statements for a contacts database. "
            "The database has one table: CONTACTS.\n\n"
            "CONTACTS Table Structure:\n"
            "- ID (INTEGER, PRIMARY KEY, AUTOINCREMENT)\n"
            "- NAME (VARCHAR, NOT NULL)\n"
            "- PHONE (INTEGER, UNIQUE, NOT NULL, 10 digits)\n"
            "- EMAIL (VARCHAR, UNIQUE, NOT NULL)\n"
            "- ADDRESS (TEXT)\n\n"
            "Rules for INSERT Statements:\n"
            "1. For CONTACTS: INSERT INTO CONTACTS (NAME, PHONE, EMAIL, ADDRESS) VALUES (...);\n"
            "2. Phone numbers must be 10-digit integers.\n"
            "3. Use single quotes for string values.\n"
            "4. Return only the SQL query, no explanations.\n\n"
            "Examples:\n"
            "1. +/(Add new) contact: INSERT INTO CONTACTS (NAME, PHONE, EMAIL, ADDRESS) VALUES ('John Doe', 5551234567, 'john@email.com', '123 Main St');\n"
            "2. Add another contact: INSERT INTO CONTACTS (NAME, PHONE, EMAIL, ADDRESS) VALUES ('Jane Smith', 9876543210, 'jane@email.com', '456 Oak Ave');"
            "3. Add task: none"
        ),
        "view": (
            "You are an expert in generating SQL SELECT queries with JOINs for a contacts and tasks database.\n\n"
            "Rules for SELECT Statements:\n"
            "1. Always use JOINs when showing tasks to include assignee names.\n"
            "2. Use LOWER() for case-insensitive comparisons in WHERE clauses.\n"
            "3. Use proper table aliases (C for CONTACTS, T for TASKS).\n"
            "4. Return only the SQL query, no explanations.\n\n"
            "Examples:\n"
            "1. Show all tasks: SELECT T.ID, T.TITLE, T.DESCRIPTION, T.CATEGORY, T.PRIORITY, T.STATUS, C.NAME AS ASSIGNEE FROM TASKS T LEFT JOIN CONTACTS C ON T.ASSIGNED_TO = C.ID;\n"
            "2. Find contacts from Delhi: SELECT * FROM CONTACTS WHERE LOWER(ADDRESS) LIKE '%delhi%';\n"
            "3. Show ongoing tasks for John: SELECT T.ID, T.TITLE, T.DEADLINE FROM TASKS T JOIN CONTACTS C ON T.ASSIGNED_TO = C.ID WHERE LOWER(C.NAME) = LOWER('John Doe') AND T.STATUS = 'In Progress';"
            "4. Display task 1: SELECT T.* FROM TASKS T JOIN CONTACTS C ON T.ASSIGNED_TO = C.ID WHERE T.ID = 1;"
        ),
        "update": (
            "You are an expert in generating SQL UPDATE statements for a contacts and tasks database.\n\n"
            "Rules for UPDATE Statements:\n"
            "1. For contacts, use ID as the identifier in WHERE clause.\n"
            "2. For tasks, use ID as the identifier in WHERE clause.\n"
            "3. Use single quotes for string values.\n"
            "4. Include only one SET clause per statement.\n"
            "5. Return only the SQL query, no explanations.\n\n"
            "Examples:\n"
            "1. Update contact email: UPDATE CONTACTS SET EMAIL = 'new@email.com' WHERE ID = 2;\n"
            "2. Mark task as completed: UPDATE TASKS SET STATUS = 'Completed' WHERE ID = 5;\n"
            "3. Change task deadline: UPDATE TASKS SET DEADLINE = '2024-12-31 23:59' WHERE ID = 3;\n"
            "4. Reassign task: UPDATE TASKS SET ASSIGNED_TO = (SELECT ID FROM CONTACTS WHERE NAME = 'vivek') WHERE ID = 10;\n"
            "5. Update contact based on name: UPDATE CONTACTS SET ADDRESS = 'New Address' WHERE LOWER(NAME) = LOWER('John Doe');\n"
            "6. Update task status based on title: UPDATE TASKS SET STATUS = 'Reviewed & Approved' WHERE LOWER(TITLE) = LOWER('Project Planning');"
        )
    }
    
    messages = [
        SystemMessage(content=system_prompts[action]),
        HumanMessage(content=prompt)
    ]
    
    try:
        response = llm.invoke(messages)
        sql_query = response.content.strip()
        
        # Clean markdown formatting if present
        if sql_query.startswith("```sql"):
            sql_query = sql_query[6:-3].strip()
        elif sql_query.startswith("```"):
            sql_query = sql_query[3:-3].strip()
            
        return sql_query
    except Exception as e:
        st.error(f"Error generating SQL query: {e}")
        return ""

def execute_query(sql_query: str, params=None):
    try:
        conn = sqlite3.connect('test.db', check_same_thread=False)
        cur = conn.cursor()
        
        if sql_query.strip().upper().startswith("SELECT"):
            cur.execute(sql_query, params or ())
            rows = cur.fetchall()
            columns = [desc[0] for desc in cur.description] if cur.description else []
            return columns, rows
        else:
            cur.execute(sql_query, params or ())
            affected_rows = cur.rowcount
            conn.commit()
            return None, affected_rows
            
    except sqlite3.Error as e:
        st.error(f"SQL error: {e}")
        return None, None
    finally:
        if conn:
            conn.close()

def classify_action(prompt: str) -> str:
    """Classify user intent into add/view/update actions using LLM."""
    system_prompt = (
        "Classify the user's database request into one of: add, view, or update. "
        "Respond ONLY with the action keyword. Rules:\n"
        "- 'add' for creating new records (insert)\n"
        "- 'view' for read operations (select)\n"
        "- 'update' for modifying existing records\n"
        "Examples:\n"
        "User: Add new contact -> add\n"
        "User: Show tasks -> view\n"
        "User: Change email -> update\n"
        "User: List contacts in NY -> view\n"
        "User: Mark task 5 completed -> update"
    )
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt)
    ]
    
    try:
        response = llm.invoke(messages)
        action = response.content.strip().lower()
        return action if action in ['add', 'view', 'update'] else 'view'
    except Exception as e:
        st.error(f"Error classifying action: {e}")
        return 'view'

def format_response(action: str, sql_query: str, rowcount: int = None, data: tuple = None):
    """Format the response based on the action."""
    responses = {
        "add": lambda: f"✅ Successfully added {rowcount} record(s)",
        "update": lambda: f"✅ Successfully updated {rowcount} record(s)",
        "view": lambda: (f"🔍 Found {len(data[1])} results:", data)
    }
    return responses[action]()

def parse_task_parameters(prompt: str) -> dict:
    """Extract task parameters from natural language input using LLM."""
    system_prompt = """Extract task parameters from user input. Return JSON with:
    - title: string
    - description: string
    - category: string (default: Work)
    - priority: string (default: Medium)
    - deadline: string (date/time in natural language)
    - assigned_to: string (contact name)
    - status: string (default: Not Started)
    Ensure valid JSON format with double quotes. Example output: 
    {"title": "Task", "priority": "High", "deadline": "tomorrow", "assigned_to": "John"}
    "input": "Need to finish client proposal ASAP",
  "output": {
    "title": "Complete Client Proposal",
    "priority": "High",
    "status": "In Progress"}}"""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt)
    ]
    
    try:
        response = llm.invoke(messages)
        parsed = json.loads(response.content.strip())
        # Ensure essential fields exist
        return {
            "title": parsed.get("title", ""),
            "priority": parsed.get("priority", "Medium"),
            "deadline": parsed.get("deadline", ""),
            "assigned_to": parsed.get("assigned_to", ""),
            "category": parsed.get("category", "Work"),
            "status": parsed.get("status", "Not Started"),
            "description": parsed.get("description", "")
        }
    except Exception as e:
        st.error(f"Parameter parsing error: {str(e)}")
        return {}

# Streamlit UI Setup
st.set_page_config(page_title="DB Manager", layout="wide")
st.sidebar.title("Navigation")

# Modify the page selection section
if 'target_page' not in st.session_state:
    st.session_state.target_page = "🏠 Home"

# Override page selection if redirected
if st.session_state.target_page != "🏠 Home":
    page = st.session_state.target_page
else:
    page = st.sidebar.radio("Go to", ["🏠 Home", "📝 New Contact", "✅ New Task", "🧩 Dashboard" , "🔍 Deep Search","📅 Gantt Chart","📄 Documents", "📊 View All Data", "🚀 PerformX",])
# Home Page
if page == "🏠 Home":
    
    if 'task_status' in st.session_state:
        st.success(st.session_state.task_status)
        del st.session_state.task_status
    
    st.header("💬 Database Chat Assistant")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Main chat logic
    if prompt := st.chat_input("What would you like to do?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        if "add task" in prompt.lower() or "create task" in prompt.lower():
            # Parse parameters and redirect to task page
            params = parse_task_parameters(prompt)
            st.session_state.prefill_task = params
            st.session_state.target_page = "✅ New Task"
            st.rerun()
        else:
            # Classify user intent
            action_type = classify_action(prompt)
        
            # Generate SQL query based on detected action
            sql_query = generate_sql_query(prompt, action_type)
        
            if sql_query:
                # st.session_state.messages.append({"role": "assistant", "content": f"Generated SQL:\n```sql\n{sql_query}\n```"})  # Debugging
                
                # Execute query
                columns, result = execute_query(sql_query)

                # Format response
                if action_type == "view" and columns:
                    response_text = format_response(action_type, sql_query, data=(columns, result))
                    df = pd.DataFrame(result, columns=columns)
                    response = f"{response_text[0]}\n\n{df.to_markdown(index=False)}"
                elif action_type in ["add", "update"]:
                    response = format_response(action_type, sql_query, rowcount=result)
                else:
                    response = "❌ No results found or invalid query"

                # Add assistant response
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()

# New Contact Page
elif page == "📝 New Contact":
    st.header("📝 Create New Contact")
    
    with st.form("contact_form", clear_on_submit=True):
        cols = st.columns(2)
        with cols[0]:
            name = st.text_input("Full Name*", help="Required field")
            phone = st.text_input("Phone Number*", max_chars=10, help="10 digits without country code")
        with cols[1]:
            email = st.text_input("Email Address*")
            address = st.text_input("Physical Address")
        
        submitted = st.form_submit_button("💾 Save Contact")
        
        if submitted:
            if not all([name, phone, email]):
                st.error("Please fill required fields (*)")
            elif not phone.isdigit() or len(phone) != 10:
                st.error("Phone must be 10 digits")
            else:
                try:
                    sql = """INSERT INTO CONTACTS (NAME, PHONE, EMAIL, ADDRESS) VALUES (?, ?, ?, ?)"""
                    params = (name.strip(), int(phone), email.strip(), address.strip())
                    _, affected = execute_query(sql, params)
                    if affected:
                        st.success("Contact created successfully!")
                        st.balloons()
                    else:
                        st.error("Error creating contact")
                except Exception as e:
                    st.error(f"Database error: {str(e)}")

# New Task Page
elif page == "✅ New Task":
    st.header("✅ Create New Task")
    
    # st.write("Debug Prefill:", st.session_state.get('prefill_task', {}))
    
    # Get contacts for assignment
    conn = sqlite3.connect('test.db', check_same_thread=False)
    cur = conn.cursor()
    cur.execute("SELECT ID, NAME FROM CONTACTS ORDER BY NAME")
    contacts = cur.fetchall()
    contact_names = [name for _, name in contacts]
    contact_dict = {name: id for id, name in contacts}
    conn.close()
    
    # Check for prefill parameters
    prefill = st.session_state.get('prefill_task', {})
    
    with st.form("task_form", clear_on_submit=True):
        # Basic Info
        col1, col2 = st.columns([2, 1])
        with col1:
            title = st.text_input("Task Title*", value=prefill.get('title', ''))
            description = st.text_area("Detailed Description", 
                                      value=prefill.get('description', ''),
                                      height=100)
        with col2:
            # Handle natural language deadlines
            deadline_input = prefill.get('deadline', '')
            default_date = datetime.today()

            if deadline_input:
                try:
                    # Use simple natural date parsing
                    if 'tomorrow' in deadline_input.lower():
                        default_date += timedelta(days=1)
                    elif 'next week' in deadline_input.lower():
                        default_date += timedelta(weeks=1)
                    elif 'in 2 days' in deadline_input.lower():
                        default_date += timedelta(days=2)
                except:
                    pass

            
            deadline_date = st.date_input("Deadline Date*", 
                                        min_value=datetime.today(),
                                        value=default_date)
            deadline_time = st.time_input("Deadline Time*", datetime.now().time())
        
        # Task Metadata
        st.subheader("Task Details", divider="rainbow")
        cols = st.columns(3)
        with cols[0]:
            category = st.selectbox("Category", ["Work", "Personal", "Project", "Other"],
                                  index=["Work", "Personal", "Project", "Other"].index(
                                      prefill.get('category', 'Work')))
            priority = st.select_slider("Priority*", options=["Low", "Medium", "High"],
                                      value=prefill.get('priority', 'Medium'))
            expected_outcome = st.text_input("Expected Outcome", 
                                           value=prefill.get('expected_outcome', ''),
                                           placeholder="e.g., Complete project setup")
        with cols[1]:
            # Safe index handling for assigned_to
            assigned_to_index = 0
            if prefill.get('assigned_to'):
                # Case-insensitive match
                lower_names = [name.lower() for name in contact_names]
                try:
                    assigned_to_index = lower_names.index(prefill['assigned_to'].lower())
                except ValueError:
                    assigned_to_index = 0
            
            assigned_to = st.selectbox("Assign To*", 
                                     options=contact_names,
                                     index=assigned_to_index)
            
            # Safe status index
            status_index = ["Not Started", "In Progress", "On Hold", "Completed"].index(
                prefill.get('status', 'Not Started'))
            status = st.selectbox("Status*", 
                                ["Not Started", "In Progress", "On Hold", "Completed"],
                                index=status_index)
            
            # Safe support contact index
            support_index = 0
            if prefill.get('support_contact'):
                try:
                    support_index = contact_names.index(prefill['support_contact'])
                except ValueError:
                    support_index = 0
            
            support_contact = st.selectbox("Support Contact", 
                                          options=contact_names,
                                          index=support_index)
        with cols[2]:
            estimated_time = st.text_input("Estimated Time", 
                                         value=prefill.get('estimated_time', ''),
                                         placeholder="e.g., 2 hours")
            required_resources = st.text_input("Required Resources",
                                             value=prefill.get('required_resources', ''))
        
        # Additional Details
        st.subheader("Additional Information", divider="rainbow")
        dependencies = st.text_area("Dependencies", value=prefill.get('dependencies', ''))
        instructions = st.text_area("Instructions", 
                                   value=prefill.get('instructions', ''),
                                   placeholder="Detailed instructions for the task")
        review_process = st.text_area("Review Process", 
                                    value=prefill.get('review_process', ''),
                                    placeholder="Steps for reviewing the task")
        performance_metrics = st.text_area("Success Metrics",
                                         value=prefill.get('performance_metrics', ''))
        notes = st.text_area("Internal Notes", value=prefill.get('notes', ''))
        
        # Proper submit button
        submitted = st.form_submit_button("🚀 Create Task")
        
        if submitted:
            if not all([title, priority, assigned_to, status, deadline_date]):
                st.error("Please fill required fields (*)")
            else:
                try:
                    deadline = datetime.combine(deadline_date, deadline_time).strftime("%Y-%m-%d %H:%M:%S")
                    sql = """INSERT INTO TASKS (
                        TITLE, DESCRIPTION, CATEGORY, PRIORITY, EXPECTED_OUTCOME,
                        DEADLINE, ASSIGNED_TO, DEPENDENCIES, REQUIRED_RESOURCES,
                        ESTIMATED_TIME, INSTRUCTIONS, REVIEW_PROCESS, PERFORMANCE_METRICS,
                        SUPPORT_CONTACT, NOTES, STATUS
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
                    
                    params = (
                        title.strip(),
                        description.strip(),
                        category,
                        priority,
                        expected_outcome.strip(),
                        deadline,
                        contact_dict[assigned_to],
                        dependencies.strip(),
                        required_resources.strip(),
                        estimated_time.strip(),
                        instructions.strip(),
                        review_process.strip(),
                        performance_metrics.strip(),
                        contact_dict.get(support_contact, None),
                        notes.strip(),
                        status
                    )
                    
                    _, affected = execute_query(sql, params)
                    if affected:
                        st.session_state.task_status = f"✅ Task '{title.strip()}' created successfully!"
                        st.session_state.target_page = "🏠 Home"
                        if 'prefill_task' in st.session_state:
                            del st.session_state.prefill_task
                        st.rerun()
                    else:
                        st.error("Error creating task")
                except Exception as e:
                    st.error(f"Database error: {str(e)}")                   

# Deep Search Page
elif page == "🔍 Deep Search":
    st.header("🔍 Deep Search with Natural Language")
    
    with st.form("deep_search_form"):
        query = st.text_area("Ask your data question:", 
                           placeholder="E.g.: Show me all high priority tasks assigned to Vivek due this week")
        
        analyze_cols = st.columns([3, 1])
        with analyze_cols[1]:
            st.markdown("### Query Tips:")
            st.markdown("""
            - Use specific filters: "tasks from last week"
            - Combine criteria: "contacts in Mumbai with email @gmail"
            - Request analysis: "average task duration by priority"
            """)
        
        submitted = st.form_submit_button("🔎 Analyze")
        
        if submitted:
            with st.spinner("Analyzing your query..."):
                try:
                    # Invoke the SQL agent
                    response = agent_executor.invoke({"input": query})
                    
                    # Display the results
                    st.subheader("Analysis Results", divider="rainbow")
                    
                    # Check if the response contains the expected output
                    if "output" in response:
                        st.markdown(f"**Result:**\n{response['output']}")
                        
                        # If the query is a SELECT, try to fetch and display the results
                        if "SELECT" in response['output'].upper():
                            conn = sqlite3.connect('test.db', check_same_thread=False)
                            cur = conn.cursor()
                            cur.execute(response['output'])
                            rows = cur.fetchall()
                            columns = [desc[0] for desc in cur.description]
                            conn.close()
                            
                            if rows:
                                df = pd.DataFrame(rows, columns=columns)
                                st.dataframe(df)
                                st.download_button(
                                    "📥 Export Results",
                                    df.to_csv(index=False),
                                    "results.csv",
                                    mime="text/csv"
                                )
                            else:
                                st.warning("No results found.")
                    else:
                        st.error("The agent did not return a valid response.")
                    
                    st.success("Analysis completed!")
                
                except Exception as e:
                    st.error(f"Search failed: {str(e)}")
                    st.markdown("**Troubleshooting Tips:**")
                    st.markdown("""
                    - Try being more specific with names/dates
                    - Use exact field names you see in forms
                    - Check for typos in contact/task names
                    """)

# Gantt Chart
elif page == "📅 Gantt Chart":
    st.header("📅 Task Timeline Visualization")
    
    @st.cache_data
    def get_tasks_for_gantt():
        try:
            conn = sqlite3.connect('test.db')
            query = """SELECT 
                TITLE, 
                DEADLINE,
                STATUS,
                PRIORITY,
                CATEGORY
                FROM TASKS"""
            df = pd.read_sql(query, conn)
            conn.close()
            
            # Convert and calculate dates
            df["DEADLINE"] = pd.to_datetime(df["DEADLINE"], errors='coerce')
            df = df.dropna(subset=["DEADLINE"])
            df["START_DATE"] = df["DEADLINE"] - pd.Timedelta(days=5)
            
            return df
        
        except sqlite3.Error as e:
            st.error(f"Database error: {str(e)}")
            return pd.DataFrame()

    tasks_df = get_tasks_for_gantt()
    
    if tasks_df.empty:
        st.warning("No tasks with valid deadlines found in the database.")
    else:
        # Create interactive filters
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_status = st.multiselect(
                "Filter by Status",
                options=tasks_df['STATUS'].unique(),
                default=tasks_df['STATUS'].unique()
            )
        with col2:
            selected_priority = st.multiselect(
                "Filter by Priority",
                options=tasks_df['PRIORITY'].unique(),
                default=tasks_df['PRIORITY'].unique()
            )
        with col3:
            selected_category = st.multiselect(
                "Filter by Category",
                options=tasks_df['CATEGORY'].unique(),
                default=tasks_df['CATEGORY'].unique()
            )

        # Apply filters
        filtered_df = tasks_df[
            (tasks_df['STATUS'].isin(selected_status)) &
            (tasks_df['PRIORITY'].isin(selected_priority)) &
            (tasks_df['CATEGORY'].isin(selected_category))
        ]

        # Create Gantt chart with Plotly
        fig = px.timeline(
            filtered_df,
            x_start="START_DATE",
            x_end="DEADLINE",
            y="TITLE",
            color="PRIORITY",
            color_discrete_map={
                "High": "#FF4B4B",
                "Medium": "#FFA500",
                "Low": "#00CC96"
            },
            title="Task Schedule Overview",
            labels={"TITLE": "Task Name"},
            hover_data=["STATUS", "CATEGORY"]
        )

        # Customize layout
        fig.update_layout(
            height=600,
            xaxis_title="Timeline",
            yaxis_title="Tasks",
            showlegend=True,
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis={'categoryorder': 'total ascending'}
        )

        # Display the chart
        st.plotly_chart(fig, use_container_width=True)
        
        # Show data summary
        st.subheader("📊 Task Statistics", divider="grey")
        cols = st.columns(4)
        cols[0].metric("Total Tasks", len(filtered_df))
        cols[1].metric("Earliest Deadline", filtered_df['DEADLINE'].min().strftime("%Y-%m-%d"))
        cols[2].metric("Latest Deadline", filtered_df['DEADLINE'].max().strftime("%Y-%m-%d"))
        cols[3].metric("Avg Duration", f"{(filtered_df['DEADLINE'] - filtered_df['START_DATE']).mean().days} days")

# All Data Page
elif page == "📊 View All Data":
    st.header("📊 View All Database Records")
    
    try:
        conn = sqlite3.connect('test.db')
        
        # Contacts Table
        st.subheader("👥 Contacts", divider="rainbow")
        df_contacts = pd.read_sql("SELECT * FROM CONTACTS ORDER BY NAME", conn)
        if not df_contacts.empty:
            st.dataframe(
                df_contacts,
                column_config={
                    "PHONE": st.column_config.NumberColumn(format="%d")
                },
                use_container_width=True
            )
            st.download_button(
                label="📥 Export Contacts",
                data=df_contacts.to_csv(index=False),
                file_name="contacts.csv",
                mime="text/csv"
            )
        else:
            st.warning("No contacts found in database")

        # Tasks Table
        st.subheader("✅ Tasks", divider="rainbow")
        df_tasks = pd.read_sql("""SELECT 
            T.ID,
            T.TITLE,
            T.DESCRIPTION,
            T.CATEGORY,
            T.PRIORITY,
            T.STATUS,
            T.DEADLINE,
            C.NAME AS ASSIGNED_TO
            FROM TASKS T
            LEFT JOIN CONTACTS C ON T.ASSIGNED_TO = C.ID
            ORDER BY DEADLINE""", conn)
        
        if not df_tasks.empty:
            st.dataframe(
                df_tasks,
                column_config={
                    "DEADLINE": st.column_config.DatetimeColumn(),
                    "PRIORITY": st.column_config.SelectboxColumn(options=["Low", "Medium", "High"])
                },
                use_container_width=True
            )
            st.download_button(
                label="📥 Export Tasks",
                data=df_tasks.to_csv(index=False),
                file_name="tasks.csv",
                mime="text/csv"
            )
        else:
            st.warning("No tasks found in database")

    except sqlite3.Error as e:
        st.error(f"Database error: {str(e)}")
    finally:
        if conn:
            conn.close()


# Resource Management Page
elif page == "🧩 Dashboard":
    st.header("🧩 Customizable Widget Dashboard", divider="rainbow")
    
    # Initialize widget configuration
    if 'dashboard_config' not in st.session_state:
        st.session_state.dashboard_config = {
            'layout': 'col2',
            'widgets': {
                'completion': True,
                'calendar': True,
                'contacts': True,
                'priority_matrix': True,
                'kpi': True
            }
        }
    
    # ===== Configuration Panel =====
    with st.expander("⚙️ Dashboard Controls", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Layout Options")
            layout = st.radio("Choose Layout", 
                ["2 Columns", "3 Columns"], 
                index=0,
                key='layout_choice'
            )
            
        with col2:
            st.subheader("Widget Toggles")
            c1, c2 = st.columns(2)
            with c1:
                st.checkbox("Task Completion", 
                    key='widget_completion', 
                    value=st.session_state.dashboard_config['widgets']['completion']
                )
                st.checkbox("Priority Matrix", 
                    key='widget_matrix', 
                    value=st.session_state.dashboard_config['widgets']['priority_matrix']
                )
            with c2:
                st.checkbox("Calendar", 
                    key='widget_calendar', 
                    value=st.session_state.dashboard_config['widgets']['calendar']
                )
                st.checkbox("Performance KPIs", 
                    key='widget_kpi', 
                    value=st.session_state.dashboard_config['widgets']['kpi']
                )
        
        with col3:
            st.subheader("Visual Settings")
            st.color_picker("Theme Color", "#2E86C1", key='dashboard_theme')
            st.slider("Widget Spacing", 1, 5, 3, key='widget_spacing')

    # ===== Widget Grid =====
    cols = st.columns(2 if layout == "2 Columns" else 3)
    col_index = 0
    
    # ===== Widget 1: Task Completion Gauge =====
    if st.session_state.widget_completion:
        with cols[col_index].container(border=True):
            st.subheader("📊 Task Completion")
            
            try:
                conn = sqlite3.connect('test.db')
                df_tasks = pd.read_sql("""
                    SELECT 
                        SUM(CASE WHEN STATUS = 'Completed' THEN 1 ELSE 0 END) as completed,
                        COUNT(*) as total
                    FROM TASKS
                """, conn)
                
                completed = df_tasks.iloc[0]['completed']
                total = df_tasks.iloc[0]['total']
                progress = int((completed/total)*100) if total > 0 else 0
                
                st.metric("Overall Progress", 
                        f"{progress}% ({completed}/{total} tasks)",
                        help="Percentage of completed tasks")
                
                # Animated progress bar
                progress_html = f"""
                <style>
                    .progress-container {{
                        background: #eee;
                        border-radius: 10px;
                        height: 20px;
                    }}
                    
                    .progress-bar {{
                        background: {st.session_state.dashboard_theme};
                        border-radius: 10px;
                        height: 100%;
                        width: {progress}%;
                        transition: width 0.5s ease-in-out;
                    }}
                </style>
                <div class="progress-container">
                    <div class="progress-bar"></div>
                </div>
                """
                st.markdown(progress_html, unsafe_allow_html=True)
                
                # Breakdown by priority
                df_priority = pd.read_sql("""
                    SELECT PRIORITY, 
                        COUNT(*) as total,
                        SUM(CASE WHEN STATUS = 'Completed' THEN 1 ELSE 0 END) as completed
                    FROM TASKS
                    GROUP BY PRIORITY
                """, conn)
                
                if not df_priority.empty:
                    st.write("### By Priority")
                    for _, row in df_priority.iterrows():
                        prio_progress = int((row['completed']/row['total'])*100) if row['total'] > 0 else 0
                        st.write(f"{row['PRIORITY']}: {prio_progress}%")
                        st.progress(prio_progress)
                
            except sqlite3.Error as e:
                st.error(f"Database error: {str(e)}")
            finally:
                conn.close()
            
        col_index += 1

    # ===== Widget 2: Calendar View =====
    if st.session_state.widget_calendar and col_index < len(cols):
        with cols[col_index].container(border=True):
            st.subheader("📅 Upcoming Deadlines")
            
            try:
                conn = sqlite3.connect('test.db')
                today = datetime.today().strftime('%Y-%m-%d')
                df_calendar = pd.read_sql(f"""
                    SELECT 
                        TITLE, 
                        DATE(DEADLINE) as deadline_date,
                        JULIANDAY(DATE(DEADLINE)) - JULIANDAY('{today}') as days_remaining
                    FROM TASKS
                    WHERE DATE(DEADLINE) >= '{today}'
                    ORDER BY DEADLINE
                    LIMIT 10
                """, conn)
                
                if not df_calendar.empty:
                    df_calendar['Due In'] = df_calendar['days_remaining'].apply(
                        lambda x: f"{int(x)} days" if x > 0 else "Today"
                    )
                    
                    # Create a styled dataframe
                    st.dataframe(
                        df_calendar[['TITLE', 'deadline_date', 'Due In']],
                        column_config={
                            "deadline_date": "Deadline",
                            "TITLE": "Task Name"
                        },
                        hide_index=True,
                        use_container_width=True
                    )
                else:
                    st.success("No upcoming deadlines! 🎉")
                
                # Add calendar heatmap
                st.write("### Deadline Heatmap")
                df_heatmap = pd.read_sql(f"""
                    SELECT 
                        DATE(DEADLINE) as date,
                        COUNT(*) as tasks
                    FROM TASKS
                    GROUP BY DATE(DEADLINE)
                """, conn)
                
                if not df_heatmap.empty:
                    fig = px.timeline(df_heatmap, 
                                    x_start="date", 
                                    x_end="date",
                                    y=[1]*len(df_heatmap),
                                    color="tasks",
                                    color_continuous_scale=[(0, "#2E86C1"), (1, "#154360")])
                    fig.update_layout(height=200, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                
            except sqlite3.Error as e:
                st.error(f"Database error: {str(e)}")
            finally:
                conn.close()
            
        col_index += 1

    # ===== Widget 3: Priority Matrix =====
    if st.session_state.widget_matrix and col_index < len(cols):
        with cols[col_index].container(border=True):
            st.subheader("📌 Priority Matrix")
            
            try:
                conn = sqlite3.connect('test.db')
                df_tasks = pd.read_sql("""
                    SELECT 
                        TITLE,
                        PRIORITY,
                        JULIANDAY(DEADLINE) - JULIANDAY('now') as days_left
                    FROM TASKS
                    WHERE STATUS != 'Completed'
                """, conn)
                
                if not df_tasks.empty:
                    # Categorize tasks
                    df_tasks['Quadrant'] = df_tasks.apply(lambda row: 
                        "🔥 Do Now" if row['PRIORITY'] == 'High' and row['days_left'] <= 3 else
                        "⏳ Schedule" if row['PRIORITY'] == 'High' else
                        "🤝 Delegate" if row['days_left'] <= 3 else
                        "🗑️ Eliminate", axis=1)
                    
                    quadrants = {
                        "🔥 Do Now": "#E74C3C",
                        "⏳ Schedule": "#2E86C1", 
                        "🤝 Delegate": "#28B463",
                        "🗑️ Eliminate": "#566573"
                    }
                    
                    # Create matrix visualization
                    fig = px.scatter(df_tasks,
                                   x=df_tasks['days_left'],
                                   y=df_tasks['PRIORITY'].map({'High':3, 'Medium':2, 'Low':1}),
                                   color='Quadrant',
                                   color_discrete_map=quadrants,
                                   hover_data=['TITLE'],
                                   labels={'x': 'Days Until Deadline', 'y': 'Priority Level'})
                    
                    fig.update_layout(
                        height=400,
                        yaxis=dict(
                            tickvals=[1, 2, 3],
                            ticktext=['Low', 'Medium', 'High']
                        )
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.success("No pending tasks! 🎉")
                
            except sqlite3.Error as e:
                st.error(f"Database error: {str(e)}")
            finally:
                conn.close()
            
        col_index += 1

    # ===== Widget 4: Performance KPIs =====
    if st.session_state.widget_kpi and col_index < len(cols):
        with cols[col_index].container(border=True):
            st.subheader("📈 Performance Metrics")
            
            try:
                conn = sqlite3.connect('test.db')
                
                # KPI 1: Weekly Completion Rate
                df_weekly = pd.read_sql("""
                    SELECT 
                        DATE(created_at, 'weekday 0', '-6 days') as week_start,
                        COUNT(*) as total,
                        SUM(CASE WHEN STATUS = 'Completed' THEN 1 ELSE 0 END) as completed
                    FROM TASKS
                    GROUP BY week_start
                    ORDER BY week_start DESC
                    LIMIT 4
                """, conn)
                
                if not df_weekly.empty:
                    latest = df_weekly.iloc[0]
                    st.metric("Weekly Completion Rate", 
                            f"{int((latest['completed']/latest['total'])*100)}%",
                            delta=f"{(latest['completed'] - df_weekly.iloc[1]['completed'])} vs last week")
                
                # KPI 2: Average Time to Complete
                df_duration = pd.read_sql("""
                    SELECT AVG(
                        JULIANDAY(completed_at) - JULIANDAY(created_at)
                    ) as avg_days
                    FROM TASKS
                    WHERE STATUS = 'Completed'
                """, conn)
                avg_days = df_duration.iloc[0]['avg_days'] or 0
                st.metric("Avg Completion Time", 
                        f"{avg_days:.1f} days",
                        help="Average time from creation to completion")
                
                # KPI 3: Overdue Tasks
                df_overdue = pd.read_sql("""
                    SELECT COUNT(*) as count
                    FROM TASKS
                    WHERE DATE(DEADLINE) < DATE('now') 
                    AND STATUS != 'Completed'
                """, conn)
                st.metric("Overdue Tasks", 
                        df_overdue.iloc[0]['count'],
                        delta_color="inverse")
                
                # Trend Visualization
                st.write("### Completion Trend")
                df_trend = pd.read_sql("""
                    SELECT 
                        DATE(completed_at) as date,
                        COUNT(*) as completed
                    FROM TASKS
                    GROUP BY DATE(completed_at)
                    ORDER BY date
                """, conn)
                
                if not df_trend.empty:
                    fig = px.line(df_trend, x='date', y='completed',
                                title="Daily Completed Tasks")
                    st.plotly_chart(fig, use_container_width=True)
                
            except sqlite3.Error as e:
                st.error(f"Database error: {str(e)}")
            finally:
                conn.close()

elif page == "📄 Documents":
    st.header("📄 Document Automation Hub")
    
    # Template selection
    doc_type = st.selectbox("Select Document Type", 
        ["Contract", "Report", "Meeting Minutes", "Custom"])
    
    # Contact selection
    conn = sqlite3.connect('test.db')
    contacts = pd.read_sql("SELECT ID, NAME FROM CONTACTS", conn)
    selected_contact = st.selectbox("Select Contact", contacts['NAME'])
    
    # Document generation
    if doc_type == "Contract":
        with st.form("contract_form"):
            st.subheader("Contract Generator")
            terms = st.text_area("Contract Terms", height=200)
            payment = st.number_input("Payment Amount", min_value=0)
            deadline = st.date_input("Contract Deadline")
            
            if st.form_submit_button("Generate Contract"):
                # Use Jinja2 template with contact details
                contract = f"""
                CONTRACT AGREEMENT
                Between {selected_contact} and Your Company
                Terms: {terms}
                Payment: ${payment}
                Deadline: {deadline}
                """
                st.download_button("Download Contract", contract, "contract.txt")
    
    # Report automation
    elif doc_type == "Report":
        st.subheader("Automated Report Generator")
        report_data = pd.read_sql("""
            SELECT T.TITLE, T.STATUS, T.DEADLINE, C.NAME 
            FROM TASKS T LEFT JOIN CONTACTS C ON T.ASSIGNED_TO = C.ID
        """, conn)
        
        if st.button("Generate Progress Report"):
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                report = report_data.to_csv(index=False)
                st.download_button("Download CSV Report", report, "task_report.csv")

elif page == "📈 Analytics":
    st.header("📈 Advanced Analytics Dashboard")
    
    # Key Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        total_tasks = get_total_tasks()
        st.metric("Total Tasks", total_tasks)
    with col2:
        overdue = get_overdue_tasks()
        st.metric("Overdue Tasks", overdue)
    with col3:
        avg_time = pd.read_sql("""
            SELECT AVG((JULIANDAY(DEADLINE) - JULIANDAY(created_at)) 
            FROM TASKS
        """, conn).iloc[0,0]
        st.metric("Avg Time to Deadline", f"{avg_time:.1f} days")
    
    # Advanced Visualizations
    tab1, tab2 = st.tabs(["Task Analysis", "Contact Insights"])
    
    with tab1:
        df_tasks = pd.read_sql("""
            SELECT STATUS, PRIORITY, CATEGORY, 
                   JULIANDAY(DEADLINE)-JULIANDAY(created_at) as duration 
            FROM TASKS
        """, conn)
        
        fig = px.box(df_tasks, x='PRIORITY', y='duration', 
                    title="Task Duration by Priority")
        st.plotly_chart(fig)
    
    with tab2:
        df_contacts = pd.read_sql("""
            SELECT ADDRESS, COUNT(*) as count 
            FROM CONTACTS 
            GROUP BY ADDRESS
        """, conn)
        
        if not df_contacts.empty:
            fig = px.density_mapbox(df_contacts, lat='lat', lon='lon', 
                                   radius=10, zoom=3, mapbox_style="open-street-map")
            st.plotly_chart(fig)

#  perform X page
elif page == "🚀 PerformX":
    st.header("🚀 Performance Tracking Dashboard")
    
    # Add meta tag for redirection with delay
    st.markdown("""
    <meta http-equiv="refresh" content="0; url='https://performtrack.streamlit.app/'">
    """, unsafe_allow_html=True)
    
    # Show loading message
    
    # Fallback link
    st.markdown("""
    <div style="text-align: center; margin-top: 2rem;">
        <h3>⏳ Redirecting to Performance Dashboard</h3>
        <p>If you're not redirected automatically, click below:</p>
        <a href="https://performtrack.streamlit.app/" target="_blank">
            <button style="
                background: #FF4B4B;
                color: white;
                padding: 1em 2em;
                border: none;
                border-radius: 10px;
                font-size: 1.2rem;
                cursor: pointer;
                margin-top: 1rem;">
                🔥 Open Performance Dashboard
            </button>
        </a>
    </div>
    """, unsafe_allow_html=True)

# Sidebar Examples Guide
st.sidebar.markdown("### Examples Guide")
st.sidebar.markdown("""
**Add Data Examples:**
- "Add new contact: John, 5551234567, john@email.com, London"
- "Create task: Project Setup, Initialize repo, 2024-12-31, 5551234567"

**View Data Examples:**
- "Show contacts from Delhi"
- "List ongoing tasks for John"
- "Display completed tasks"

**Update Data Examples:**
- "Change John's email to new@email.com"
- "Mark task 5 as completed"
- "Update task 3's due date to tomorrow"
""")

if st.button("Push Database Changes to GitHub"):
    print('hello')