import time, math
import numpy as np
import pandas as pd
import xgboost
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

import streamlit as st
import SessionState

session_state = SessionState.get(train_button=False,
                                 has_trained_before=False,
                                 tested_males=False,
                                 tested_females=False)


sd_h1 = st.sidebar.header("Training Options")
sd_show_code = st.sidebar.checkbox("Show code", False)
sd_include_sensitive_data = st.sidebar.checkbox('Include sensitive data', True)

sd_h2 = st.sidebar.subheader("Hyperparameters")
sd_train_test_split = st.sidebar.slider('Train-test split', 0.8, 0.9, 0.8, 0.05)
sd_n_estimators = st.sidebar.slider('Number of estimators',
                                    min_value=10,
                                    max_value=100,
                                    value=50,
                                    step=10)
sd_max_depth = st.sidebar.slider('Max depth', 3, 10, 5)

sd_h1 = st.sidebar.header("Oversampling Options")
sd_n_bootstrap = st.sidebar.slider('Size of oversampled groups', 100, 1000, 200)
sd_boostrap_replace = st.sidebar.checkbox('Sample with replacement', True)

NON_SENSITIVE_COLS = ['ave_miles_driven_daily', 'debt', 'monthly_income']

st.title('The Accuracy-Fairness Dilemma')
st.text("Machine learning models merely amplify our biases - not eliminate them.")
st.markdown("by [Franz Louis Cesista](https://twitter.com/leloykun)")

st.markdown('<hr size="4">', unsafe_allow_html=True)
st.write("Newbie data scientists tend to put the accuracy of their models on a pedestal. Couple this with their disdain of the social sciences and they end up automating discrimination instead of fighting it.")
st.write("")
st.write("I admit I was guilty of this too. \"If we just replace flawed humans with cold, calculating machines,\" I thought, \"then we can eliminate discrimination in the world.\" Seems reasonable, right? But, this view is naive. Machines can be flawed too and their creators don't have to be evil for them to be so.")
st.write("")
st.write("Fairness doesn't follow from the accuracy of our models. In fact, the two are inherently conflicted. This is what I call the Accuracy-Fairness Dilemma:")
st.markdown("> To maximize accuracy, models have to learn everything they can from the data - including the human biases embedded in them. But to maximize fairness, they have to unlearn the human biases.")
st.write("")
st.write("We want to teach machines as much as we can, but we may also end up teaching them the mistakes of the past.")

st.header('A Concrete Example')

st.write("")
st.write("Car salespeople have this uncanny ability to predict how much their customers are really able to pay for a car, despite the latter's denials. It takes years of practice to do this well. But, wouldn't it be nice if we could build a machine learning model that can do the same? With it, we could rake in a lot more profits with much less experience under our belt.")
st.write("")
st.write("Here we have historical data on car sales of a certain car retailer. The first five columns contain customer data while the last column contains how much customers paid for a car.")

df_car_sales = pd.read_csv('car_sales_data.csv')
if sd_include_sensitive_data:
    df_display_table = df_car_sales.copy()
else:
    df_display_table = df_car_sales[NON_SENSITIVE_COLS + ['$_sales']].copy()
st.table(df_display_table.sort_values(by=['$_sales'], ascending=False, ignore_index=True).head())

st.write("A newbie data scientist would just blindly maximize the accuracy of their models without taking ethical and social considerations into account. To simulate that process, just click the button below")

col1, col2, col3 = st.beta_columns(3)
with col2:
    train_button = st.button("Train", key='train')

train_code = '''train_features = df_car_sales.sample(frac={})
test_features = df_car_sales.drop(train_features.index)

train_labels = train_features.pop('$_sales')
test_labels = test_features.pop('$_sales')

xgb = XGBRegressor(verbose=0,
                   n_estimators={},
                   max_depth={})
xgb.fit(train_features, train_labels)'''.format(sd_train_test_split, sd_n_estimators, sd_max_depth)
if sd_show_code:
    st.code(train_code, language='python')

@st.cache
def train_test_split(df_car_sales):
    train_features = df_car_sales.sample(frac=sd_train_test_split)
    test_features = df_car_sales.drop(train_features.index)
    train_labels = train_features.pop('$_sales')
    test_labels = test_features.pop('$_sales')
    return (train_features, train_labels), (test_features, test_labels)

@st.cache(hash_funcs={'xgboost.sklearn.XGBRegressor': id})
def train_model(features, labels, include_sensitive_data):
    xgb = XGBRegressor(verbose=0,
                       n_estimators=sd_n_estimators,
                       max_depth=sd_max_depth,
                       eval_metric='rmse',
                       objective="reg:squarederror")
    if include_sensitive_data:
        xgb.fit(features, labels)
    else:
        xgb.fit(features[:,2:], labels)
    return xgb

def score_model(model, test_features, test_labels, include_sensitive_data):
    if include_sensitive_data:
        test_preds = model.predict(test_features)
    else:
        test_preds = model.predict(test_features[:,2:])
    return math.sqrt(mean_squared_error(test_preds, test_labels))

def calc_pay_gap(model, features):
    male_features   = features[features['gender'] == 0].sample(n=sd_n_bootstrap, replace=sd_boostrap_replace, random_state=42)
    female_features = features[features['gender'] == 1].sample(n=sd_n_bootstrap, replace=sd_boostrap_replace, random_state=42)
    if not sd_include_sensitive_data:
        male_features   = male_features[NON_SENSITIVE_COLS]
        female_features = female_features[NON_SENSITIVE_COLS]
    return (model.predict(female_features.values) - model.predict(male_features.values)).mean()

if train_button:
    session_state.train_button = True

if session_state.train_button:
    if not session_state.has_trained_before:
        session_state.has_trained_before = True
        # for a little bit of flair
        progress_bar = st.progress(0)
        status_text = st.empty()
        for i in range(1, 10+1):
            progress_bar.progress(i*10)
            status_text.text("%i%% Complete" % (10*i))
            time.sleep(0.1)
        progress_bar.empty()
        st.balloons()

    (train_features, train_labels), (test_features, test_labels) = train_test_split(df_car_sales)
    xgb = train_model(train_features.values, train_labels.values, sd_include_sensitive_data)
    rmse = score_model(xgb, test_features.values, test_labels.values, sd_include_sensitive_data)

    st.write("Yay! You just trained your first machine learning model! The expected prediction error of this model is around ${:.3f}. This isn't a lot when we consider how expensive the cars are (see the table above).".format(rmse))

    h1_s1 = st.subheader("Try it out Yourself:")
    c_age = st.number_input('Age', value=21, min_value=18, max_value=130)
    c_gender = st.selectbox('Gender', ('Male', 'Female'))
    c_driven = st.number_input('Average miles driven daily', value=30)
    c_debt = st.number_input('Debt', value=0, step=500)
    c_monthly_income = st.number_input('Monthly Income (in $)', value=7000, step=500)

    features = [
        c_driven,
        c_debt,
        c_monthly_income
    ]
    if sd_include_sensitive_data:
        features = [
            c_age,
            1*(c_gender == 'Female')
        ] + features
    print(features)
    features = np.array(features).reshape((1,-1))
    pred = xgb.predict(features)[0]

    st.markdown("**Prediction:**")
    st.info("This customer is willing to pay ${:.2f} for a car.".format(pred))

    if c_gender == 'Male':
        session_state.tested_males = True
    else:
        session_state.tested_females = True

    h1_s2 = st.subheader('How to Measure Unfairness')

    if session_state.tested_males and session_state.tested_females:
        pay_gap = calc_pay_gap(xgb, test_features)
        more_payer_group = 'female'
        less_payer_group = 'male'
        if pay_gap < 0:
            more_payer_group, less_payer_group = less_payer_group, more_payer_group
            pay_gap *= -1
        st.write("If you deployed this model to production, it would've misled your company into charging {} customers ${:.2f} more than the {} for no logical reason at all. That would be unethical and discriminatory.".format(more_payer_group, pay_gap, less_payer_group))
        st.write("")
        st.write("I know anecdotes aren’t enough to prove theories. Thus, we need a more mathematical approach to show that our model really is discriminatory.")
        st.write("")
        st.write("There are a lot of valid ways to measure unfairness, but I followed the steps below for this demo:")
        st.markdown("1. Partition the dataset into two groups\n2. Oversample the groups to make their sizes equal; and\n3. Use the following formula to calculate the unfairness of the model:")
        st.code("Unfairness = E[prediction(Group A) - prediction(Group B)]")
        st.write("")
        st.markdown("Notice that if `unfairness = 0`, then we can say our model treats female and male customers equally. In this case, `unfairness = {:.2f}`. This means that the model recommends us to charge females ${:.2f} more than males, on average.".format(pay_gap, pay_gap))

        if sd_show_code:
            st.code('''male_features   = test_features[test_features['gender'] == 0].sample(n={}, replace={})
female_features = test_features[test_features['gender'] == 1].sample(n={}, replace={})
unfairness = (model.predict(female_features.values) - model.predict(male_features.values)).mean()
'''.format(sd_n_bootstrap, sd_boostrap_replace, sd_n_bootstrap, sd_boostrap_replace, NON_SENSITIVE_COLS, NON_SENSITIVE_COLS),
            language='python')
    else:
        st.write("**(You need to compare the predictions for males and females first before we could continue...)**")
else:
    st.write("**(You need to train your model first before we could continue...)**")


st.markdown('<hr size="5">', unsafe_allow_html=True)
st.header('Conclusion')
st.write("")
st.write("Machine learning models merely amplify our biases - not eliminate them.")
st.write("")
st.write("The Accuracy-Fairness Dilemma also generalizes to all models, even our mental models of the world. We have a lot of incentives to make them as accurate as possible. But, we can’t just blindly optimize their accuracy because we may also end up teaching them the mistakes of the past.")

st.markdown('<hr size="5">', unsafe_allow_html=True)
st.write("In the next demo, we will explore how to make machine learning models fairer. We will also throw privacy considerations into the mix and explore the Accuracy-Fairness-Privacy Trilemma. If you don't want to miss out, please subscribe to my newsletter!")
st.markdown("<iframe src=\"https://ponder.substack.com/embed\" width=\"100%\" height=\"320\" style=\"border:1px solid #EEE; background:white;\" frameborder=\"0\" scrolling=\"no\"></iframe>", unsafe_allow_html=True)


st.markdown('<hr size="5">', unsafe_allow_html=True)
st.header('FAQs')
st.subheader("Why don’t we prevent the model from accessing sensitive data while training?")
st.write("Uncheck 'Include sensitive data' in the sidebar to the left and redo the demo. I can guarantee the unfairness wouldn't be eliminated.")

st.subheader("What if the data isn’t \"clean\" in the first place?")
st.write("You shouldn’t fudge with or delete someone's data just because they seem \"odd\" along with everybody else's. We are unique in our own way and our data reflect that. It is what it is and you just have to deal with it.")
