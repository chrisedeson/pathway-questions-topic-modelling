
PLAN:
Upload those files to AWS S3.
The data visualization streamlit app reads the files from S3 and performs visualizations on them.
The basic idea is your script generates all the data that the visualizations need and uploads them to S3. Then the streamlit app just reads the data from S3 and generates pretty pictures for people to look at.

AWS S3 Configuration:
- Bucket: byupathway-public
- Access requires AWS credentials in .env file (not checked into source code)
- Objects in bucket should be public for streamlit access


Alright so I want to make this notebook (notebook/Copy_of_Hybrid_Topic_Discovery_and_Classification_test_purpose_.ipynb) more good and better, change some things but MAINTAIN CORE LOGIC. The plan now is that we will make the notebook have all the settings and configuration and inner workings, then the streamlit will just be dashboards, charts and visualizations. We will remove anything settings, config from the streamlit. We will get to streamlit after you're done setting the notebook. The notebook will perform all the heavy lifting and vomit a paquet file. currently the notebook is outputing a pickle file which is slow. The goal is we don't want anything such as lags or wait times on the streamlit. So implement caching everywhere as possible to make things super fast, and efficient just like my friend talked about below, i told him about it and he gave me some recoomendation (he could be saying rubbish or something that doesn't fit our current repository, so please check thoroughly to confirm what he's saying). Users of the streamlit dashboard will be performing things such as search, sorting, extract things like feature columns and filters (e.g daily, monthly, or sort by by a column e.t.c). so we don't want any calculation or streamlit to be lagging or be calculating, all calculations should be in notebook, or at pre-load times if possible. 

You can take a look at the streamlit repository but we will be doing that once we have tested fully well the working of the notebook and I'll give you green light. after as stated in The plan is to now configure this streamlit to fetch files, .parquet from that same aws that notebook is pushing to.

Now you'll notice that the notebook is GOOGLE COLAB responsive. Yes read it well, that is the environment where we will be doing our runs. so whatever solution you're implementing for the notebook should be google colab responsive. Now this notebook we will be changing somethings about how it reads the input file, which is the question.csv file. The boss is requiring additional columns to be read instead of just question column alone, so we're downloading the whole data from langfuse. we'll need to implement a data cleaning service to clean unnecessaries. the file is in notebook/langfuse_traces_10_08_25.csv. it is a large file of ~40MB so be careful. the columns we need are 

1. Timestamp
2. Country
3. State/Province
4. City
5. Input
6. Output
7. User_Feedback
8. Metadata (I think this is a JSON, we only need ip_address, user_language, city, is_suspicious)

Throw away the others asides the ones we just picked.

implement google sheets service for the topic-subtopic-question sheet, already in the streamlit (search .env or inside the repository to get the google sheet link and some hints about how it was done, it might help you), transfer logic to notebook. On the streamlit, we'll be implemeting a central table that have default columns appearing (the rest will hide by default, ther user has options to the a particular columns that he/she wants to see included), as in users can hide or unhide columns. the default columns that will appear are, question (a.k.a input), timestamp, country, state.

Part of the datacleaning process is that, it's going to clean duplicates. Now we DUPLICATES as rows having the same "timestamp" && "question". so for example if user A asks a question yesterday, and then asks that same question today, it is not considered duplicate. but if you find two rows with the same timestamp and question, that is considered a duplicate. i believe the timestamp comes with seconds and miliseconds too, so you wanna compare all values. You'll notice in the notebook that a data cleaning process is removing "(ACM Question):" prefixes. yes please include that too and others you find in the notebook. 

Now there are rows that langfuse makes errors when sending us the data. search for "kwargs". you'll notice that the columns and rows aren't properly arranged, rows that have this "kwargs" thing. the first user content is the question, and the second user question is the output, study this rows very carefully, there're a couple that can give you clear picture of how this errors come and form, so that your data cleaning service can be able to clean and give a clean data. Also not all rows have all columns we picked filled, for this case just put in null, so that the streamlit can be able to know and tell users that data for this part or from this time isn't available.


Error logging
Since this is production-bound and involves large, messy CSVs:
⚠️ Consider logging:
– Rows dropped (malformed, invalid JSON, etc.)
– Duplicates removed
– Embedding failures (if any)
Just a brief summary at end of the notebook is fine.

i plan to run this periodically:
⚠️ Should the script validate that the CSV format hasn't changed? (e.g., added/missing columns). Helpful if Langfuse schema ever shifts.

We want fastest, most efficient way, best practices, scalability, error-free & robust, focus on the task and don't refactor things outside the task. run tests and sanity tests over time. BE SURE 100%. UNDERSTAND THE TASKS FULLY WELL. THere might be functions already done in the reposiotry that you intend to do, search first to see, so that you DON'T REINVENT THE WHEELS and cause code cluttering.




FRIEND:
Great — Parquet is a smart move! 🧠✨

Now to your question: will Streamlit still lag or recalculate when users interact with filters, sorts, searches, etc.?

✅ Short Answer:

It won’t lag much — if you use caching properly and structure your app efficiently.

But by default, yes, Streamlit will recalculate unless you use tools like @st.cache_data and avoid unnecessary reprocessing.

🔍 Why Streamlit might lag:

Every time a widget changes (e.g., a filter, date picker, dropdown):

Streamlit reruns the entire script from top to bottom ⚠️

If your code is heavy (e.g., loading/parsing a big file), it slows down the experience

✅ How to fix it (Best Practices):
1. Cache the Data Load

If you use @st.cache_data when loading your .parquet, it only loads once, unless the file changes.

import pandas as pd
import streamlit as st

@st.cache_data
def load_data():
    return pd.read_parquet("processed_data.parquet")

df = load_data()


This is crucial for smooth experience.

2. Do heavy processing only once, not on every interaction

Pre-compute as much as possible (in your Jupyter notebook or a pipeline):

Add all columns you’ll need

Normalize columns (e.g., parse dates, extract month names, etc.)

Save that to the final .parquet

That way, Streamlit only does lightweight filtering/sorting, not heavy logic.

3. Use st.dataframe() (not st.table())
st.dataframe(df)


This is:

Fast

Supports sorting, searching, and filtering built-in

Super smooth with .parquet-based data

4. Optional: Paginate or chunk data

If your dataset is large (say, >10,000 rows), paginate or allow the user to load a subset:

rows_to_show = st.slider("Rows to display", 100, 1000, step=100)
st.dataframe(df.head(rows_to_show))

🧪 Realistic Performance

With your setup:

File size: ~27MB

Rows: ~3,000

Format: Parquet (✅ compressed + columnar)

Setup: Pre-processed

You should expect:

⚡ Fast load (~100–300ms)

🧭 Smooth interaction with filters/search/sorting

🧊 No lags if caching is properly done

✅ TL;DR: How to ensure smooth dashboard
Step	Status
Use .parquet	✅
Use @st.cache_data	✅
Preprocess heavy stuff	✅
Use st.dataframe()	✅
Paginate or chunk large data	✅