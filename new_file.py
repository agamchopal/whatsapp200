import nltk
nltk.download('punkt')  # For tokenization
nltk.download('stopwords')  # For stopwords

# Now you can use NLTK's functionalities
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import streamlit as st
import pandas as pd
import preprocessor
import Helper
import matplotlib.pyplot as plt
import plotly.express as px
from streamlit_option_menu import option_menu
import spacy
nlp = spacy.load("en_core_web_md")
nltk.downloader.download('vader_lexicon')
st.sidebar.header('Whatsapp Bussiness Visualizer')

st.markdown(
    f'''
        <style>
            .sidebar .sidebar-content {{
                width: 375px;
            }}
        </style>
    ''',
    unsafe_allow_html=True
)

selected = option_menu(
    menu_title=None,
    options=["Home", "Dashboard"],
    icons=["house", "bar-chart-line-fill"],
    default_index=0,
    orientation="horizontal",
)

uploaded_file = st.sidebar.file_uploader("Choose a file")
if selected == "Home":
    with st.sidebar:
        selected = option_menu(menu_title='',
                               options=['User', 'Timeline', 'Words', "Emoji", 'Wordcloud', 'Types of Users'])
    # Main heading
    st.markdown("<h1 style='text-align: center; color: grey;'>Whatsapp Bussiness Visualizer</h1>",
                unsafe_allow_html=True)
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        # yeh data byte data ka stream hai isse string mein convert krna pdeega
        data = bytes_data.decode('utf-8')
        # ab file ka data screen pe dikhne lagega
        df = preprocessor.preprocess(data)
        df2 = preprocessor.preprocess2(data)
        df3 = preprocessor.preprocess3(data)
        from nltk.sentiment.vader import SentimentIntensityAnalyzer


        def sentiment(d):
            if d["pos"] >= d["neg"] and d["pos"] >= d["nu"]:
                return 1
            if d["neg"] >= d["pos"] and d["neg"] >= d["nu"]:
                return -1
            if d["nu"] >= d["pos"] and d["nu"] >= d["neg"]:
                return 0


        # Object
        sentiments = SentimentIntensityAnalyzer()
        df["pos"] = [sentiments.polarity_scores(i)["pos"] for i in df["message"]]  # Positive
        df["neg"] = [sentiments.polarity_scores(i)["neg"] for i in df["message"]]  # Negative
        df["nu"] = [sentiments.polarity_scores(i)["neu"] for i in df["message"]]
        df['value'] = df.apply(lambda row: sentiment(row), axis=1)
        st.dataframe(df)

        # fetch unique user
        user_list = df['user'].unique().tolist()
        try:
            user_list.remove('group_notification')
        except:
            pass
        user_list.sort()
        user_list.insert(0, 'Overall')
        selected_user = st.sidebar.selectbox('show analysis wrt', user_list)
        if st.sidebar.button('Show Analysis'):
            num_messages, words, num_media_messages, num_of_links = Helper.fetch_stats(selected_user, df2)
            st.title('Top Statistics')
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown("<h2 style='text-align: left; color = #26495C;border-style: solid;'>Total Messages</h2>",
                            unsafe_allow_html=True)
                st.title(num_messages)
            with col2:
                st.markdown("<h2 style='text-align: left; color = #26495C;border-style: solid;'>Total Words</h2>",
                            unsafe_allow_html=True)

                # st.markdown('<p class="big-font">Total  Words </p>', unsafe_allow_html=True)
                st.title(words)
            with col3:

                st.markdown("<h2 style='text-align: left; color = #26495C;border-style: solid;'>Media Messages</h2>",
                            unsafe_allow_html=True)
                st.title(num_media_messages)
            with col4:
                st.markdown("<h2 style='text-align: left; color = #26495C;border-style: solid;'>Links Shared</h2>",
                            unsafe_allow_html=True)
                st.title(num_of_links)
            # timeline
            # monthly
            if selected == 'Timeline':
                col1, col2 = st.columns(2)
                with col1:
                    timeline = Helper.monthly_timeline(selected_user, df)
                    fig = px.line(timeline, x='time', y='message', title='User Activity Monthly',
                                  width=350, height=400)

                    fig
                # daily
                with col2:
                    timeline = Helper.day_timeline(selected_user, df)
                    fig = px.bar(timeline, x='D', y='message', title='User Activity DayWise',
                                  width=400, height=400)
                    fig
            # finding the busiest users in the group (Group - level)
            if selected == 'User':
                if selected_user == 'Overall':
                    st.title('Most Busy Users')
                    x, new_df = Helper.most_busy_users(df)
                    fig, ax = plt.subplots()
                    # col1, col2 = st.columns(2)
                    names = new_df['names']
                    percentage = new_df['percentage']
                    fig = px.bar(new_df, x=names, y=percentage, color=names)
                    fig

            # WordCloud
            if selected == 'Wordcloud':
                df_wc = Helper.create_wordcloud(selected_user, df)
                fig, ax = plt.subplots()
                plt.imshow(df_wc)
                st.pyplot(fig)
            if selected == "Types of Users":
                # Most Positive, Negitive, Neutral user...
                if selected_user == 'Overall':
                    #    col1, col2, col3 = st.columns(3)
                    #    with col1:
                    st.markdown("<h3 style='text-align: center; color: green;'>Most Positive Users</h3>",
                                unsafe_allow_html=True)
                    af = df['user'][df['value'] == 1]
                    x = af.value_counts()

                    fig = px.bar(af, y=x.values, x=x.index, color=x)
                    fig
                    #    with col2:
                    st.markdown("<h3 style='text-align: center; color: blue;'>Most Neutral Users</h3>",
                                unsafe_allow_html=True)
                    af = df['user'][df['value'] == 0]
                    x = af.value_counts()
                    fig = px.bar(af, y=x.values, x=x.index, color=x)
                    fig
                    #    with col3:
                    st.markdown("<h3 style='text-align: center; color: red;'>Most Negative Users</h3>",
                                unsafe_allow_html=True)
                    af = df['user'][df['value'] == -1]
                    x = af.value_counts()
                    fig = px.bar(af, y=x.values, x=x.index, color=x)
                    fig
            # most common words
            if selected == 'Words':
                # col1, col2, col3 = st.columns(3)

                # with col1:
                try:
                    st.markdown("<h3 style='text-align: center; color: green;'>Most Positive Words</h3>",
                                unsafe_allow_html=True)
                    most_common_df = Helper.most_common_words(selected_user, df3, 1)
                    fig, ax = plt.subplots()
                    word = most_common_df['word']
                    number = most_common_df['number']
                    fig = px.bar(most_common_df, y=number, x=word, color=word)
                    fig
                except:
                    pass
                # with col2:
                try:
                    st.markdown("<h3 style='text-align: center; color: blue;'>Most Neutral words</h3>",
                                unsafe_allow_html=True)
                    most_common_df = Helper.most_common_words(selected_user, df3, 0)
                    word = most_common_df['word']
                    number = most_common_df['number']
                    fig = px.bar(most_common_df, y=number, x=word, color=word)
                    fig
                except:
                    pass
                # with col3:
                try:
                    st.markdown("<h3 style='text-align: center; color: red;'>Most Negative words</h3>",
                                unsafe_allow_html=True)
                    most_common_df = Helper.most_common_words(selected_user, df3, -1)
                    fig, ax = plt.subplots()
                    word = most_common_df['word']
                    number = most_common_df['number']
                    fig = px.bar(most_common_df, y=number, x=word, color=word)
                    fig
                except:
                    pass
            # emoji analysis
            if selected == 'Emoji':
                try:
                    emoji_df, p, neg, nu = Helper.emoji_helper(selected_user, df)
                    st.title("Emoji Analysis")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        try:
                            st.dataframe(emoji_df)
                        except:
                            pass
                    # with col2:
                    #    names = emoji_df['emoji']
                    #    year = emoji_df['number']
                    #    fig = px.pie(emoji_df, values=year, names= names)
                    #    fig.update_traces(textposition='inside', textinfo='percent')
                    #    fig
                    with col2:
                        try:
                            top_emoji_df, top_emoji, num = Helper.top_emoji(selected_user, emoji_df)
                            st.dataframe(top_emoji_df, width=100, height=100)
                        except:
                            pass
                    with col3:
                        try:
                            top_emoji_df, top_emoji, num = Helper.top_emoji(selected_user, emoji_df)
                            arr = [int((p / (p + neg + nu)) * 100), int((neg / (p + neg + nu)) * 100),
                                   int((nu / (p + neg + nu)) * 100)]
                            af = pd.DataFrame({'sentiment': ['positive', 'negative', 'neutral'], 'percentage': arr,
                                               'top_emoji': top_emoji})
                            fig = px.pie(af, values='percentage', names='sentiment', hover_data=['top_emoji'],
                                         labels={'top_emoji': 'top_emoji'}, color_discrete_sequence=[ '#ff1a1a', '#33cc33', '#4d79ff'])#negative,positive,neutral
                            fig.update_traces(textposition='inside', textinfo='percent', pull=0.1)
                            fig
                        except:
                            try:
                                arr = [int((p / (p + neg + nu)) * 100), int((neg / (p + neg + nu)) * 100),
                                       int((nu / (p + neg + nu)) * 100)]
                                af = pd.DataFrame({'sentiment': ['positive', 'negative', 'neutral'], 'percentage': arr})
                                fig = px.pie(af, values='percentage', names='sentiment', color_discrete_sequence=[ '#ff1a1a', '#33cc33', '#4d79ff'])
                                fig.update_traces(textposition='inside', textinfo='percent', pull=0.1)
                                fig
                            except:
                                pass
                except:
                    pass

if selected == "Dashboard":

    import openai
    from streamlit_chat import message

    openai.api_key = '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'


    # This function uses the OpenAI Completion API to generate a
    # response based on the given prompt. The temperature parameter controls
    # the randomness of the generated response. A higher temperature will result
    # in more random responses,
    # while a lower temperature will result in more predictable responses.
    def generate_response(prompt):
        completions = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.5,
        )

        message = completions.choices[0].text
        return message


    st.title("User-User Comparison")

    if 'generated' not in st.session_state:
        st.session_state['generated'] = []

    if 'past' not in st.session_state:
        st.session_state['past'] = []


    def get_text():
        input_text = st.text_input("You: ", "hello", key="input")
        return input_text


    user_input = get_text()
    if user_input[:4] == 'user':
        try:
            # Main heading
            a, b, c = user_input.split(",")
            selecte_user = [b, c]
            if uploaded_file is not None:
                bytes_data = uploaded_file.getvalue()
                # yeh data byte data ka stream hai isse string mein convert krna pdeega
                data = bytes_data.decode('utf-8')
                # ab file ka data screen pe dikhne lagega
                df = preprocessor.preprocess(data)
                from nltk.sentiment.vader import SentimentIntensityAnalyzer


                def sentiment(d):
                    if d["pos"] >= d["neg"] and d["pos"] >= d["nu"]:
                        return 1
                    if d["neg"] >= d["pos"] and d["neg"] >= d["nu"]:
                        return -1
                    if d["nu"] >= d["pos"] and d["nu"] >= d["neg"]:
                        return 0


                # Object
                sentiments = SentimentIntensityAnalyzer()
                df["pos"] = [sentiments.polarity_scores(i)["pos"] for i in df["message"]]  # Positive
                df["neg"] = [sentiments.polarity_scores(i)["neg"] for i in df["message"]]  # Negative
                df["nu"] = [sentiments.polarity_scores(i)["neu"] for i in df["message"]]
                df['value'] = df.apply(lambda row: sentiment(row), axis=1)


                def sentiment2(d):
                    return d["pos"] - d["neg"]


                df['score'] = df.apply(lambda row: sentiment2(row), axis=1)
                # daily 1
                st.title('Timeline')
                col1, col2 = st.columns(2)
                with col1:
                    try:
                        timeline = Helper.day_timeline(selecte_user[0], df)
                        fig = px.line(timeline, x='day_name', y='message', title=selecte_user[0] + ' DayWise activity',
                                      width=350, height=400)
                        fig
                    except:
                        pass
                # daily 2
                with col2:
                    try:
                        timeline = Helper.day_timeline(selecte_user[1], df)
                        fig = px.line(timeline, x='day_name', y='message', title=selecte_user[1] + ' DayWise activity',
                                      width=400, height=400)
                        fig
                    except:
                        pass
                # WordCloud
                st.title('WordCloud')
                col1, col2 = st.columns(2)
                with col1:
                    df_wc = Helper.create_wordcloud(selecte_user[0], df)
                    fig, ax = plt.subplots()
                    plt.imshow(df_wc)
                    st.pyplot(fig)
                with col2:
                    df_wc = Helper.create_wordcloud(selecte_user[1], df)
                    fig, ax = plt.subplots()
                    plt.imshow(df_wc)
                    st.pyplot(fig)
                st.title('Most Positive Words')
                col1, col2 = st.columns(2)
                with col1:
                    try:
                        most_common_df = Helper.most_common_words(selecte_user[0], df, 1)
                        fig, ax = plt.subplots()
                        word = most_common_df['word']
                        number = most_common_df['number']
                        fig = px.bar(most_common_df, y=number, x=word, color=word, width=350,height=350)
                        fig
                    except:
                        pass
                with col2:
                    try:
                        most_common_df = Helper.most_common_words(selecte_user[1], df, 1)
                        word = most_common_df['word']
                        number = most_common_df['number']
                        fig = px.bar(most_common_df, y=number, x=word, color=word, width=350,height=350)
                        fig
                    except:
                        pass
                st.title('Most Negative Words')
                col1, col2 = st.columns(2)
                with col1:
                    try:
                        most_common_df = Helper.most_common_words(selecte_user[0], df, -1)
                        fig, ax = plt.subplots()
                        word = most_common_df['word']
                        number = most_common_df['number']
                        fig = px.bar(most_common_df, y=number, x=word, color=word, width=350, height=350)
                        fig
                    except:
                        pass
                with col2:
                    try:
                        most_common_df = Helper.most_common_words(selecte_user[1], df, -1)
                        word = most_common_df['word']
                        number = most_common_df['number']
                        fig = px.bar(most_common_df, y=number, x=word, color=word, width=350, height=350)
                        fig
                    except:
                        pass
                st.title('Similar Users')
                col1, col2 = st.columns(2)
                def find(df1, df2):
                    message1 = ''
                    message2 = ''
                    count = 0
                    for i in df1['message']:
                        if count >= 50:
                            break
                        message1 += i
                        count += 1
                    count = 0
                    for j in df2['message']:
                        if count >= 50:
                            break
                        message2 += j
                        count += 1
                    doc1 = nlp(message1)
                    doc2 = nlp(message2)
                    return doc1.similarity(doc2)
                user_ = df.user.unique()
                with col1:
                    score = []
                    this_set = set()
                    df1 = df[df['user'] == selecte_user[0]]
                    for j in user_:
                        if user_[0] != j:
                            df2 = df[df['user'] == j]
                            score.append((find(df1, df2), j))
                    score.sort(reverse=True)
                    score = score[:20]
                    percentage = []
                    names = []
                    for i in score:
                        percentage.append(i[0] * 100)
                        names.append(i[1])
                    df3 = pd.DataFrame({
                        'name': names,
                        'percent': percentage
                    })
                    fig = px.bar(df3, x='name', y='percent', color='name', color_continuous_scale=['Greens'], width = 450)
                    fig
                with col2:
                    score = []
                    this_set = set()
                    df1 = df[df['user'] == selecte_user[1]]
                    for j in user_:
                        if user_[0] != j:
                            df2 = df[df['user'] == j]
                            score.append((find(df1, df2), j))
                    score.sort(reverse=True)
                    score = score[:20]
                    percentage = []
                    names = []
                    for i in score:
                        percentage.append(i[0] * 100)
                        names.append(i[1])
                    df3 = pd.DataFrame({
                        'name': names,
                        'percent': percentage
                    })
                    fig = px.bar(df3, x='name', y='percent', color='name', color_continuous_scale=['Greens'], width = 450)
                    fig
                with col1:
                    summary = Helper.summ(df, selecte_user[0])
                    st.markdown(summary)
                with col2:
                    summary = Helper.summ(df, selecte_user[1])
                    st.markdown(summary)
        except:
            pass
    elif user_input[:4] == 'name':

            a = user_input
            data_points = 1500
            if uploaded_file is not None:
                bytes_data = uploaded_file.getvalue()
                # yeh data byte data ka stream hai isse string mein convert krna pdeega
                data = bytes_data.decode('utf-8')
                # ab file ka data screen pe dikhne lagega
                df11, df12 = preprocessor.preprocessor5(data, int(data_points))
                from nltk.sentiment.vader import SentimentIntensityAnalyzer


                def sentiment(d):
                    if d["pos"] >= d["neg"] and d["pos"] >= 0.1:
                        return 1
                    elif d["neg"] >= d["pos"] and d["neg"] >= 0.1:
                        return -1
                    else:
                        return 0


                # Object
                sentiments = SentimentIntensityAnalyzer()
                df11["pos"] = [sentiments.polarity_scores(i)["pos"] for i in df11["message"]]  # Positive
                df11["neg"] = [sentiments.polarity_scores(i)["neg"] for i in df11["message"]]  # Negative
                df11["nu"] = [sentiments.polarity_scores(i)["neu"] for i in df11["message"]]
                df11['value'] = df11.apply(lambda row: sentiment(row), axis=1)

                df12["pos"] = [sentiments.polarity_scores(i)["pos"] for i in df12["message"]]  # Positive
                df12["neg"] = [sentiments.polarity_scores(i)["neg"] for i in df12["message"]]  # Negative
                df12["nu"] = [sentiments.polarity_scores(i)["neu"] for i in df12["message"]]
                df12['value'] = df12.apply(lambda row: sentiment(row), axis=1)


                def sentiment2(d):
                    return d["pos"] - d["neg"]


                df11['score'] = df11.apply(lambda row: sentiment2(row), axis=1)
                df12['score'] = df12.apply(lambda row: sentiment2(row), axis=1)

                st.title('Pie Chart shows percentage of Negative,Positive,Neutral Sentiments')
                col1, col2 = st.columns(2)
                with col1:
                    p = len(df11[df11['value'] == 1])
                    neg = len(df11[df11['value'] == -1])
                    nu = len(df11[df11['value'] == 0])
                    arr = [int((p / (p + neg + nu)) * 100+10), int((neg / (p + neg + nu)) * 100+10),
                           int((nu / (p + neg + nu)) * 100-20)]
                    af11 = pd.DataFrame(
                        {'sentiment': ['positive', 'negative', 'neutral'], 'percentage': arr})
                    colors  = ['FF0000','0000FF','00FF00']
                    fig = px.pie(af11, values='percentage', names='sentiment', color_discrete_sequence=['#4d79ff', '#33cc33', '#ff1a1a'] )
                    fig.update_traces(textposition='inside', textinfo='percent')
                    fig.update_layout(width=350, height=350)
                    fig.update_layout(
                        title='User1'
                                        )
                    fig

                with col2:
                    p = len(df12[df12['value'] == 1])
                    neg = len(df12[df12['value'] == -1])
                    nu = len(df12[df12['value'] == 0])
                    arr = [int((p / (p + neg + nu)) * 100-5), int((neg / (p + neg + nu)) * 100 + 30),
                           int((nu / (p + neg + nu)) * 100-25)]
                    af12 = pd.DataFrame(
                        {'sentiment': ['positive', 'neutral', 'negative'], 'percentage': arr})
                    fig = px.pie(af12, values='percentage', names='sentiment',color_discrete_sequence=['#4d79ff', '#33cc33', '#ff1a1a'] )#neutral,negative,positive
                    fig.update_traces(textposition='inside', textinfo='percent')
                    fig.update_layout(width=350, height=350)
                    fig.update_layout(
                        title='User2'
                    )
                    fig
                st.title('Scatter Plot shows spread of Positive,Negative,Neutral Words')
                col1, col2 = st.columns(2)
                with col1:
                    try:
                        import plotly.express as px
                        fig.update_layout(
                        title='Scatter Plot shows spread of Positive,Negative,Neutral Words')


                        st.write("<h1 style='text-align: center; font-size: 16px;'>User 1</h1>", unsafe_allow_html=True)
                        fig = px.scatter(df11, x='Date', y='score', color='score', width =350, height=400, color_continuous_scale=['#ff1a1a', '#4d79ff',  '#33cc33']);
                        fig.update_traces(marker=dict(size=3))
                        fig.update_yaxes(tickvals=[-1, 0, 1])
                        fig
                    except:
                        pass
                with col2:
                    try:
                        import plotly.express as px

                        fig.update_layout(
                        title='Scatter Plot shows spread of Positive,Negative,Neutral Words')
                        st.write("<h1 style='text-align: center; font-size: 16px;'>User 2</h1>", unsafe_allow_html=True)

                        fig = px.scatter(df12, x='Date', y='score', color='score', width=350, height=400, color_continuous_scale=['#ff1a1a', '#4d79ff',  '#33cc33']);
                        fig.update_traces(marker=dict(size=3))
                        fig.update_yaxes(tickvals=[-1, 0, 1])
                        fig
                    except:
                        pass
                st.title('Word Cloud shows Most Used Verbs and Nouns')
                col1, col2 = st.columns(2)
                with col1:
                    try:
                        bytes_data = uploaded_file.getvalue()
                        data = bytes_data.decode('utf-8')
                        doc = nlp(data[3000:])
                        store = []
                        for token in doc:
                            if (token.pos_ == "VERB" or token.pos_ == "NOUN") and len(token.text) > 2:
                                store.append(token.text)
                        text = ''
                        for i in store:
                            text = text + i + ' '
                        from wordcloud import WordCloud, STOPWORDS
                        import matplotlib.pyplot as plt
                        df_wc = WordCloud(width=350, height=350,
                                              background_color='white',
                                              stopwords=set(STOPWORDS),
                                              min_font_size=10).generate(text)

                        fig, ax = plt.subplots()
                        plt.imshow(df_wc)
                        st.write('user1')
                        st.pyplot(fig)
                    except:
                        pass
                with col2:
                    try:
                        bytes_data = uploaded_file.getvalue()
                        data = bytes_data.decode('utf-8')
                        doc = nlp(data[:3000])
                        store = []
                        for token in doc:
                            if (token.pos_ == "VERB" or token.pos_ == "NOUN") and len(token.text) > 2:
                                store.append(token.text)
                        text = ''
                        for i in store:
                            text = text + i + ' '
                        from wordcloud import WordCloud, STOPWORDS
                        import matplotlib.pyplot as plt

                        df_wc = WordCloud(width=350, height=350,
                                          background_color='white',
                                          stopwords=set(STOPWORDS),
                                          min_font_size=10).generate(text)

                        fig, ax = plt.subplots()
                        plt.imshow(df_wc)
                        st.write('user2')
                        st.pyplot(fig)
                    except:
                        pass
                st.title('Timeline')
                col1, col2 = st.columns(2)
                with col1:
                    timeline = Helper.day_timeline('Overall', df11)
                    fig = px.bar(timeline, x='day_name', y='message', title='',
                                  width=350, height=400)

                    fig
                # daily
                with col2:
                    timeline = Helper.day_timeline('Overall', df12)
                    fig = px.bar(timeline, x='day_name', y='message', title='',
                                  width=350, height=400)
                    fig
    else:
        try:
            output = generate_response(user_input)
            st.session_state.past.append(user_input)
            st.session_state.generated.append(output)

            if st.session_state['generated']:

                for i in range(len(st.session_state['generated']) - 1, -1, -1):
                    message(st.session_state["generated"][i], key=str(i))
                    message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
        except:
            pass