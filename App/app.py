import streamlit as st 
import altair as alt
import pandas as pd 
import numpy as np 
import base64
import sklearn 
import joblib 
pipe_lr = joblib.load(open("models/emotion_classifier_pipe_lr.pkl","rb"))

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_png_as_page_bg(png_file):
    bin_str = get_base64(png_file) 
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    min-height: 1vh;
    align-items: top;
    background-size: cover;
    background-repeat: repeat;
    background-attachment: scroll; #doesn't work
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

set_png_as_page_bg('/Users/laksh/OneDrive/Desktop/emotion.png')

def predict(docx):
	results = pipe_lr.predict([docx])
	return results[0]

def probabilities(docx):
	results = pipe_lr.predict_proba([docx])
	return results

def main():
	st.title("Emotion Detection App")
	st.subheader("Classifies and detects emotion in text")

	with st.form(key='emotion_clf_form'):
			raw_text = st.text_area("Type Text Here")
			submit_text = st.form_submit_button(label='Submit')

	if submit_text:
			col1,col2  = st.columns(2)

			prediction = predict(raw_text)
			probability = probabilities(raw_text)
			
			with col1:
				
				st.success("Prediction")
				
				st.write("{}".format(prediction))
				st.write("Confidence Percentage: {}".format((np.max(probability))*100))



			with col2:
				st.success("Prediction Probability")
				probability_df = pd.DataFrame(probability,columns=pipe_lr.classes_)
				probability_df_clean = probability_df.T.reset_index()
				probability_df_clean.columns = ["emotions","probability"]

				fig = alt.Chart(probability_df_clean).mark_bar().encode(x='emotions',y='probability',color='emotions')
				st.altair_chart(fig,use_container_width=True)
                
if __name__ == '__main__':
	main()