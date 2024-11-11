import streamlit as st

# Initialize the state for the buttons if they do not exist
if 'lstm_expanded' not in st.session_state:
    st.session_state['lstm_expanded'] = False
if 'transformer_expanded' not in st.session_state:
    st.session_state['transformer_expanded'] = False
if 'traditional_expanded' not in st.session_state:
    st.session_state['traditional_expanded'] = False
if 'llma_expanded' not in st.session_state:
    st.session_state['llma_expanded'] = False

# lstm button
if 'data_preprocessing_expanded' not in st.session_state:
    st.session_state['data_preprocessing_expanded'] = False
if 'word_embedding_expanded' not in st.session_state:
    st.session_state['word_embedding_expanded'] = False
if 'model_training_expanded' not in st.session_state:
    st.session_state['model_training_expanded'] = False
if 'results_expanded' not in st.session_state:
    st.session_state['results_expanded'] = False

# final button
if 'accuracy_expanded' not in st.session_state:
    st.session_state['accuracy_expanded'] = False
if 'precision_expanded' not in st.session_state:
    st.session_state['precision_expanded'] = False
if 'recall_expanded' not in st.session_state:
    st.session_state['recall_expanded'] = False    
if 'f1_expanded' not in st.session_state:
    st.session_state['f1_expanded'] = False   

# llama button
if 'llama_data_preprocessing_expanded' not in st.session_state:
    st.session_state['llama_data_preprocessing_expanded'] = False 
if 'llama_model_loading_expanded' not in st.session_state:
    st.session_state['llama_model_loading_expanded'] = False 
if 'llama_optimized_lora_expanded' not in st.session_state:
    st.session_state['llama_optimized_lora_expanded'] = False 
if 'llama_data_loading_expanded' not in st.session_state:
    st.session_state['llama_data_loading_expanded'] = False 
if 'llama_model_training_expanded' not in st.session_state:
    st.session_state['llama_model_training_expanded'] = False 
if 'llama_predict_expanded' not in st.session_state:
    st.session_state['llama_predict_expanded'] = False 
if 'llama_confusion_matrix_expanded' not in st.session_state:
    st.session_state['llama_confusion_matrix_expanded'] = False 
if 'llama_results_expanded' not in st.session_state:
    st.session_state['llama_results_expanded'] = False 

def toggle_state(state_name):
    st.session_state[state_name] = not st.session_state[state_name]

st.title("Final Project - Sentiment Analysis")
st.markdown("Group Clusters: A13")
st.markdown("Group Members: ")

# briefly introduction

st.markdown("#### Introduction")

st.markdown("<hr style='border: 0.5px dashed black;'>", unsafe_allow_html=True)

st.markdown("#### Methods")

# Methods Buttons in a single row
methods_cols = st.columns([1, 1, 1, 1])

with methods_cols[1]:
    if st.button("LSTM"):
        toggle_state('lstm_expanded')
with methods_cols[2]:
    if st.button("Transformer"):
        toggle_state('transformer_expanded')
with methods_cols[0]:
    if st.button("Traditional"):
        toggle_state('traditional_expanded')
with methods_cols[3]:
    if st.button("Llama"):
        toggle_state('llma_expanded')

# Display LSTM content if expanded
if st.session_state['lstm_expanded']:
    st.write("##### LSTM Introduction")
    st.write("- Long Short-Term Memory(LSTM) is a type of recurrent neural network(RNN) used for sequence prediction problems.")
    st.image('images/lstm_figure.png')
    st.write('- LSTMs incorporate unique gating mechanisms, including forget, input, and output gates, which allow them to regulate the flow of information and avoid the vanishing gradient problem common in standard RNNs.')
    st.write('- It is capable of learning long-term dependencies and is effective for time-series data.')
    
    if st.button("Data Preprocessing"):
        toggle_state('data_preprocessing_expanded')
    if st.session_state['data_preprocessing_expanded']:
        st.markdown("- Remove special characters")
        st.image('images/lstm_clean_review.png')
        st.markdown("- Stopwords Removal")
        st.markdown("- lemmatization")
        st.image('images/lstm_lemmatization.png')


    if st.button("Word Embedding"):
        toggle_state('word_embedding_expanded')
    if st.session_state['word_embedding_expanded']:
        st.markdown("##### Word2Vec")
        st.markdown("- Mode: Skip-gram")
        st.image('images/word_embedding.png')
        st.image('images/lstm_word2vec_pre.png')
        st.markdown("- max_length=200")
        st.markdown("- vectors dimensionality:100")
        st.markdown("- padding mechanism")


    if st.button("Model Training"):
        toggle_state('model_training_expanded')
    if st.session_state['model_training_expanded']:
        st.markdown("##### Model: Bi-Directional LSTM")
        st.markdown('- Bi-Directional RNNs consist of two separate RNNs: one moving forward through the input sequence and the other moving backward.')
        st.image('images/model_training.png')
        st.image('images/lstm_compile.png')
        st.markdown('- Loss Function: BinaryCrossentropy')
        st.markdown('- Metrics: accuracy')
        st.image('images/callback.png')
        st.markdown('- Callback: EarlyStopping, ReduceLOnPlateau')
        st.markdown('- Model fit parameters: epochs, batch_size')
        st.image('images/modelfit.png')
    
    if st.button("Results"):
        toggle_state('results_expanded')
    if st.session_state['results_expanded']:
        
        st.markdown('##### Accuracy Convergence plot on different datasets')
        st.markdown('###### IMDB')
        st.markdown('- val_accuracy: <span style="color: red;">**0.87**</span>', unsafe_allow_html=True)
        st.markdown('- Stopping occurred at the 10th epoch.', unsafe_allow_html=True)
        st.image('images/results_lstm_imdb.png')

        st.markdown('###### Twitter')
        st.markdown('- val_accuracy: <span style="color: red;">**0.72**</span>', unsafe_allow_html=True)
        st.markdown('- Early stopping occurred at the 9th epoch, with the best result achieved at the <span style="color: red;">**6th**</span> epoch.', unsafe_allow_html=True)
        st.markdown('- ReduceLROnPlateau reducing learning rate to 2.499999936844688e-05 at 8th epoch')
        st.image('images/results_lstm_twitter.png')
        
        st.markdown('###### Google Play')
        st.markdown('- val_accuracy: <span style="color: red;">**0.75**</span>', unsafe_allow_html=True)
        st.markdown('- Early stopping occurred at the 9th epoch, with the best result achieved at the <span style="color: red;">**6th**</span> epoch.', unsafe_allow_html=True)
        st.markdown('- ReduceLROnPlateau reducing learning rate to 2.499999936844688e-05 at 8th epoch')
        st.image('images/results_lstm_google.png')


        st.markdown('##### Results on different datasets')
        st.markdown("""
            <table style='width:100%; border: 0.5px solid black; border-collapse: collapse;'>
            <tr>
                <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> </td>
                <td style='border: 0.5px solid black; padding: 10px; width: 150px;'>Accuracy</td>
                <td style='border: 0.5px solid black; padding: 10px; width: 150px;'>Precision</td>
                <td style='border: 0.5px solid black; padding: 10px; width: 150px;'>Recall</td>
                <td style='border: 0.5px solid black; padding: 10px; width: 150px;'>F1-score</td>
            </tr>
            <tr>
                <td style='border: 0.5px solid black; padding: 10px; width: 200px;'> IMDB</td>
                <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 87%</td>
                <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 84%</td>
                <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 90%</td>
                <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 87%</td>
            </tr>
            <tr>
                <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> Twitter</td>
                <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 72%</td>
                <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 71%</td>
                <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 71%</td>
                <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 71%</td>
            </tr>
            <tr>
                <td style='border: 0.5px solid black; padding: 10px; width: 250px;'> GooglePlay</td>
                <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 75%</td>
                <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 82%</td>
                <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 75%</td>
                <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 78%</td>
            </tr>
            </table>
            """, unsafe_allow_html=True)
        
        st.markdown(
                    """
                    <style>
                    .justified-text {
                        text-align: justify;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )

        # 使用 CSS 类的文本
        st.markdown("<div class='justified-text'>1) On the IMDB dataset, the model performed best. The Recall reached 90%, indicating that the model is highly effective in identifying positive (or target) classes, with a low miss rate. Overall, the F1-score on the IMDB dataset (87%) is the highest, showing a good balance between Precision and Recall.</div>", unsafe_allow_html=True)
        st.markdown("<div class='justified-text'>2) On the Twitter dataset, the model's performance was weaker. Both Precision and Recall were only 71%, resulting in an F1-score of 71%. This suggests that on the Twitter dataset, the model faces some challenges, likely due to its limited adaptability to noise and content diversity.</div>", unsafe_allow_html=True)
        st.markdown("<div class='justified-text'>3) On the GooglePlay dataset, the model performed relatively well. Precision reached 82%, indicating a relatively high confidence in the samples predicted as positive.</div>", unsafe_allow_html=True)
        st.markdown("<div class='justified-text'>4) The model's performance on the Twitter and Google Play datasets is relatively poor, possibly due to informal language, abbreviations, emojis, and differences in text length, which are not fully consistent with the IMDB data. The lack of targeted adjustments for these differences may have led to insufficient generalization capability of the model.</div>", unsafe_allow_html=True)



# Display Transformer content if expanded
if st.session_state['transformer_expanded']:
    st.write("### Transformer Options")

# Display Traditional content if expanded
if st.session_state['traditional_expanded']:
    st.write("### Traditional Options")

# Display LLMA content if expanded
if st.session_state['llma_expanded']:
    st.write("##### Llama Introduction")
    st.image('images/llama/1.png')

    st.markdown(
                        """
                        <style>
                        .justified-text {
                            text-align: justify;
                        }
                        </style>
                        """,
                        unsafe_allow_html=True
                    )
    st.markdown("<div class='justified-text'>Llama 3.2 3B is an advanced language model developed as part of the Llama (Large Language Model Meta AI) series. This model, with 3 billion parameters, is optimized for a variety of NLP tasks, such as text generation, language understanding, and dialogue. So in this project we choose to use Llma3.2 as the base model to perform sentiment analysis task. Its architecture allows for efficient processing and enhanced context handling, making it suitable for applications requiring high-quality language comprehension while maintaining manageable computational requirements.</div>", unsafe_allow_html=True)
    st.image('images/llama/2.png')
    st.markdown("<div class='justified-text'>LoRA (Low-Rank Adaptation) is a lightweight fine-tuning technique designed to reduce the cost of fine-tuning large language models. It works by freezing most of the model's parameters and only updating a small set of parameters in low-rank matrices, significantly reducing memory usage and computational demands. LoRA is especially useful in resource-constrained environments, enabling efficient fine-tuning of large models without compromising performance. In order to train Llama on premise, We choose to leverage LoRA.</div>", unsafe_allow_html=True)
    st.markdown(' ')
    
    if st.button("Data Preprocessing"):
        toggle_state('llama_data_preprocessing_expanded')
    if st.session_state['llama_data_preprocessing_expanded']:
        st.markdown("- In order to transform original csv file into the format that LLM could read.")
        st.image('images/llama/3.jpg')
    
    if st.button("Model Loading"):
        toggle_state('llama_model_loading_expanded')
    if st.session_state['llama_model_loading_expanded']:
        st.image('images/llama/4.png')
    
    if st.button("Optimized Lora Parameter"):
        toggle_state('llama_optimized_lora_expanded')
    if st.session_state['llama_optimized_lora_expanded']:
        st.image('images/llama/5.jpg')
    
    if st.button("Data Loading"):
        toggle_state('llama_data_loading_expanded')
    if st.session_state['llama_data_loading_expanded']:
        st.image('images/llama/6.jpg')

    if st.button("Model Training"):
        toggle_state('llama_model_training_expanded')
    if st.session_state['llama_model_training_expanded']:
        st.image('images/llama/7.jpg')

    if st.button("Predict Test Data"):
        toggle_state('llama_predict_expanded')
    if st.session_state['llama_predict_expanded']:
        st.image('images/llama/8.jpg')
    
    if st.button("Confusion Matrix Calculation"):
        toggle_state('llama_confusion_matrix_expanded')
    if st.session_state['llama_confusion_matrix_expanded']:
        st.image('images/llama/9.jpg')

    if st.button("Results on different datasets"):
        toggle_state('llama_results_expanded')
    if st.session_state['llama_results_expanded']:
        
        st.markdown("""
            <table style='width:100%; border: 0.5px solid black; border-collapse: collapse;'>
            <tr>
                <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> </td>
                <td style='border: 0.5px solid black; padding: 10px; width: 150px;'>Accuracy</td>
                <td style='border: 0.5px solid black; padding: 10px; width: 150px;'>Precision</td>
                <td style='border: 0.5px solid black; padding: 10px; width: 150px;'>Recall</td>
                <td style='border: 0.5px solid black; padding: 10px; width: 150px;'>F1-score</td>
            </tr>
            <tr>
                <td style='border: 0.5px solid black; padding: 10px; width: 200px;'> IMDB</td>
                <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 94.78%</td>
                <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 94.80%</td>
                <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 94.78%</td>
                <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 94.78%</td>
            </tr>
            <tr>
                <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> Twitter</td>
                <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 55.99%</td>
                <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 54.19%</td>
                <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 61.51%</td>
                <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 57.62%</td>
            </tr>
            <tr>
                <td style='border: 0.5px solid black; padding: 10px; width: 250px;'> GooglePlay</td>
                <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 74.24%</td>
                <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 76.53%</td>
                <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 83.53%</td>
                <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 79.88%</td>
            </tr>
            </table>
            """, unsafe_allow_html=True)
        
        st.markdown(
                    """
                    <style>
                    .justified-text {
                        text-align: justify;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )

        # 使用 CSS 类的文本
        st.markdown("<div class='justified-text'>From the above data, it is evident that the performance of the LLM varies significantly across different datasets, which may be due to the diversity of the datasets and the adaptability of the model.</div>", unsafe_allow_html=True)
        st.markdown(' ')
        st.markdown("<div class='justified-text'>1) IMDB Dataset: The model performs well on the IMDB dataset, achieving nearly 95% accuracy, with Precision, Recall, and F1-score almost equal. This suggests that the model effectively recognizes sentiment categories. This may be because the IMDB dataset focuses on movie reviews, which usually contain clear expressions of sentiment, making it easier for the model to learn sentiment features.</div>", unsafe_allow_html=True)
        st.markdown(' ')
        st.markdown(
            "<div class='justified-text'>"
            "2) Twitter Dataset: The performance on the Twitter dataset is noticeably lower, "
            "especially in terms of accuracy and Precision, although Recall is slightly higher. "
            "This may be because Twitter data often includes short phrases, informal language, "
            "abbreviations, misspellings, and even sarcasm or humor, presenting a complex sentiment expression. "
            "This increases the demand on the model's understanding capabilities, resulting in poorer classification performance. "
            "Additionally, sentiment classification on Twitter data may be more subjective, with more 'ambiguous' or 'neutral' emotions, "
            "which could make it more challenging for the model to distinguish positive and negative sentiments."
            "</div>",
            unsafe_allow_html=True
        )
        st.markdown(' ')
        st.markdown(
            "<div class='justified-text'>"
            "3) Google Play Dataset: The model's performance on the Google Play dataset falls between that of IMDB and Twitter. "
            "This dataset generally consists of user reviews on apps, including feedback on product features and user experience. "
            "The sentiment in such reviews is relatively clear, but some reviews may contain mixed sentiments (e.g., 'Good features, but a bit expensive'), "
            "which may impact classification accuracy. Therefore, the model shows good Recall and F1-score, though Precision is slightly lower."
            "</div>",
            unsafe_allow_html=True
        )  



st.markdown("<hr style='border: 0.5px dashed black;'>", unsafe_allow_html=True)

# final button
st.markdown('#### Final Results')

# Final Results Buttons in a single row
final_results_cols = st.columns([1, 1, 1, 1])

with final_results_cols[0]:
    if st.button('Accuracy'):
        toggle_state('accuracy_expanded')
with final_results_cols[1]:
    if st.button('Precision'):
        toggle_state('precision_expanded')
with final_results_cols[2]:
    if st.button('Recall'):
        toggle_state('recall_expanded')
with final_results_cols[3]:
    if st.button('F1-score'):
        toggle_state('f1_expanded')

# accuracy
if st.session_state['accuracy_expanded']:
    st.markdown("""
        <table style='width:100%; border: 0.5px solid black; border-collapse: collapse;'>
        <tr>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> </td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'>Traditional</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'>LSTM</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'>Transformer</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'>llma3.2</td>
        </tr>
        <tr>
            <td style='border: 0.5px solid black; padding: 10px; width: 200px;'> Accuarcy on IMDB</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> </td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 87%</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 85%</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 90%</td>
        </tr>
        <tr>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> Accuarcy on Twitter</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> </td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 72%</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> </td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> </td>
        </tr>
        <tr>
            <td style='border: 0.5px solid black; padding: 10px; width: 250px;'> Accuarcy on GooglePlay</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> </td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 75%</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> </td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> </td>
        </tr>
        </table>
        """, unsafe_allow_html=True)

# precision
if st.session_state['precision_expanded']:  
    st.markdown("""
        <table style='width:100%; border: 0.5px solid black; border-collapse: collapse;'>
        <tr>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> </td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'>Traditional</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'>LSTM</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'>Transformer</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'>llma3.2</td>
        </tr>
        <tr>
            <td style='border: 0.5px solid black; padding: 10px; width: 200px;'> Accuarcy on IMDB</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> </td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 84%</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> </td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> </td>
        </tr>
        <tr>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> Accuarcy on Twitter</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> </td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 71%</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> </td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> </td>
        </tr>
        <tr>
            <td style='border: 0.5px solid black; padding: 10px; width: 250px;'> Accuarcy on GooglePlay</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> </td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 82%</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> </td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> </td>
        </tr>
        </table>
        """, unsafe_allow_html=True)

# recall
if st.session_state['recall_expanded']:  
    st.markdown("""
        <table style='width:100%; border: 0.5px solid black; border-collapse: collapse;'>
        <tr>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> </td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'>Traditional</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'>LSTM</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'>Transformer</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'>llma3.2</td>
        </tr>
        <tr>
            <td style='border: 0.5px solid black; padding: 10px; width: 200px;'> Accuarcy on IMDB</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> </td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 90%</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> </td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> </td>
        </tr>
        <tr>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> Accuarcy on Twitter</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> </td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 71%</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> </td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> </td>
        </tr>
        <tr>
            <td style='border: 0.5px solid black; padding: 10px; width: 250px;'> Accuarcy on GooglePlay</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> </td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 75%</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> </td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> </td>
        </tr>
        </table>
        """, unsafe_allow_html=True)

# f1-score
if st.session_state['f1_expanded']:  
    st.markdown("""
        <table style='width:100%; border: 0.5px solid black; border-collapse: collapse;'>
        <tr>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> </td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'>Traditional</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'>LSTM</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'>Transformer</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'>llma3.2</td>
        </tr>
        <tr>
            <td style='border: 0.5px solid black; padding: 10px; width: 200px;'> Accuarcy on IMDB</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> </td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 87%</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> </td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> </td>
        </tr>
        <tr>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> Accuarcy on Twitter</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> </td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 71%</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> </td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> </td>
        </tr>
        <tr>
            <td style='border: 0.5px solid black; padding: 10px; width: 250px;'> Accuarcy on GooglePlay</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> </td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 78%</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> </td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> </td>
        </tr>
        </table>
        """, unsafe_allow_html=True)