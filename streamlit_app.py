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

sd_h2 = st.sidebar.header("Hyperparameters")
sd_train_test_split = st.sidebar.slider('Train-test split', 0.8, 0.9, 0.8, 0.05)
sd_n_estimators = st.sidebar.slider('Number of estimators',
                                    min_value=10,
                                    max_value=100,
                                    value=50,
                                    step=10)
sd_max_depth = st.sidebar.slider('Max depth', 3, 10, 5)
sd_n_bootstrap = st.sidebar.slider('Number of Bootstrapped Samples', 100, 1000, 200)
sd_boostrap_replace = st.sidebar.checkbox('Bootstrap with Replacement', True)

NON_SENSITIVE_COLS = ['ave_miles_driven_daily', 'debt', 'monthly_income']

st.title('The Accuracy-Fairness Dilemma - A Demo')
st.write("Machine learning models merely amplify our biases - not eliminate them.")

st.markdown('<hr size="3">', unsafe_allow_html=True)
p1 = st.write("Newbie data scientists tend to put the accuracy of their models on a pedestal. Couple this with their disdain of the social sciences, and they end up automating discrimination instead of fighting it.")
st.write("")
p2 = st.write("I admit I was guilty of this too. \"If we just replace flawed humans with cold, calculating machines,\" I thought, \"then we can eliminate discrimination in the world.\" Seems tempting, right? But, this view is naive. Machines can be flawed too and their creators don't have to be evil for them to be so.")
st.write("")
p3 = st.write("Fairness don't follow from the accuracy of our models. In fact, they're inherently conflicted. This is what I call the Accuracy-Fairness Dilemma:")
q1 = st.markdown("> To maximize accuracy, models have to learn everything they can from the data - including the human biases embedded in it. But to maximize fairness, they have to _unlearn_ the human biases.")
st.write("")

h1 = st.header('A Concrete Example')

st.write("")
p5 = st.write("Car salespeople have this uncanny ability to predict how much their customers are really able to pay for a car, despite the latter's denials. It takes _years_ of practice to do this well. But, wouldn't it be nice if we build a machine learning model that can do the same? With it, we could rake in a lot more profits with much less experience under our belt.")
st.write("")
p6 = st.write("Here we have historical data of car sales of a certain car retailer. The first five columns contain customer data while the last column contains how much customers paid for a car.")

df_car_sales = pd.read_csv('car_sales_data.csv')
if sd_include_sensitive_data:
    df_display_table = df_car_sales.copy()
else:
    df_display_table = df_car_sales[NON_SENSITIVE_COLS + ['$_sales']].copy()
st.table(df_display_table.sort_values(by=['$_sales'], ascending=False, ignore_index=True).head())

p7 = st.write("A newbie data scientist would just blindly maximize the accuracy of their models without taking ethical and social considerations into account. To simulate that process, just click the button below")

col1, col2, col3 = st.beta_columns(3)
with col2:
    train_button = st.button("Train", key='train')

train_code = '''train_features = df_car_sales.sample(frac={}, random_state=0)
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
    train_features = df_car_sales.sample(frac=sd_train_test_split, random_state=0)
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
                       objective="reg:squarederror",
                       random_state=0)
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

    p8 = st.write("Yay! You just trained your first machine learning model! The expected prediction error of this model is around ${:.3f}. This isn't a lot when we consider how expensive the cars are (see the table above).".format(rmse))

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
    st.info("This customer is willing to pay ${:.3f} for a car.".format(pred))

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
        st.write("Have you noticed that the model expects {}s to pay more for cars than {}s? In fact, the predicted sales gap is around ${:.3f}.".format(more_payer_group, less_payer_group, pay_gap))
        st.write("")
        st.write("If you deployed this model, it would've misled your company to charge {} customers ${:.3f} more than the {}s. That would be unethical and discriminatory.".format(more_payer_group, pay_gap, less_payer_group))
        st.write("")
        st.write("For the more seasoned readers, here's how you can measure the unfairness of your models. First, partition your dataset into two groups. Then, bootstrap the groups by oversampling them. For best results, make sure that the bootstrapped samples have the same cardinality. And, finally, use the following formula to calculate the unfairness of your model:")
        st.code("Unfairness = E[prediction(Group A) - prediction(Group B)]")

        if sd_show_code:
            st.code('''male_features   = test_features[test_features['gender'] == 0].sample(n={}, replace={})
female_features = test_features[test_features['gender'] == 1].sample(n={}, replace={})
unfairness = (model.predict(female_features.values) - model.predict(male_features.values)).mean()
'''.format(sd_n_bootstrap, sd_boostrap_replace, sd_n_bootstrap, sd_boostrap_replace, NON_SENSITIVE_COLS, NON_SENSITIVE_COLS),
            language='python')
    else:
        st.write("Try to compare the predictions for males and females. Which group does the model recommend us to charge more?")

        p9 = st.write("**(You need to compare the predictions for males and females first before we could continue...)**")
else:
    p7 = st.write("**(You need to train your model first before we could continue...)**")


st.markdown('<hr size="5">', unsafe_allow_html=True)
st.header('Conclusion')
st.write("")
st.write("Machine learning models merely amplify our biases - not eliminate them.")
st.write("")
st.write("The Accuracy-Fairness Dilemma generalizes too: *all* models, even our mental models of the world, can be unfair if we just blindly optimize its accuracy. Yes, the past and the present sucks. But we shouldn't just give up and say, \"it's how the world works and there's nothing we can do about it.\" We *can* change the world for the better.")

st.markdown('<hr size="5">', unsafe_allow_html=True)
st.write("In the next demo, we will explore how to make machine learning models more fair. We will also throw in privacy considerations into the mix and explore the Accuracy-Fairness-Privacy Trilemma. If you don't want to miss out, please subscribe to my newsletter below:")
st.markdown("<iframe src=\"https://ponder.substack.com/embed\" width=\"100%\" height=\"320\" style=\"border:1px solid #EEE; background:white;\" frameborder=\"0\" scrolling=\"no\"></iframe>", unsafe_allow_html=True)


st.markdown('<hr size="5">', unsafe_allow_html=True)
st.header('FAQs')
st.subheader("What if we don't feed sensitive data to the model while training it?")
st.write("Uncheck 'Include sensitive data' in the sidebar to the left and redo the experiment. I can guarantee the pay gap or unfairness wouldn't be eliminated.")

st.subheader("What if we just clean up the data?")
st.write("You can't just fudge with or delete someone's data just because they seem \"odd\" along with everybody else's. We are unique in our own way and our data reflect that. It is what it is and you just have to deal with it.")
