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

# svm button
if 'svm_data_preprocessing_expanded' not in st.session_state:
    st.session_state['svm_data_preprocessing_expanded'] = False 
if 'svm_text_processing_expanded' not in st.session_state:
    st.session_state['svm_text_processing_expanded'] = False 
if 'svm_training_model_expanded' not in st.session_state:
    st.session_state['svm_training_model_expanded'] = False 
if 'svm_results_expanded' not in st.session_state:
    st.session_state['svm_results_expanded'] = False 

# transformer button
if 'transformer_data_preprocessing_expanded' not in st.session_state:
    st.session_state['transformer_data_preprocessing_expanded'] = False 
if 'transformer_model_expanded' not in st.session_state:
    st.session_state['transformer_model_expanded'] = False 
if 'transformer_train_test_expanded' not in st.session_state:
    st.session_state['transformer_train_test_expanded'] = False 
if 'transformer_results_expanded' not in st.session_state:
    st.session_state['transformer_results_expanded'] = False 

def toggle_state(state_name):
    st.session_state[state_name] = not st.session_state[state_name]

st.title("Final Project - Sentiment Analysis")
st.markdown("Group Clusters: A13")
st.markdown("Group Members: LIU MIAO, SHI XULUN, ZHOU HANCHENG, HE JUNXIAN, YANG LIPENG")

# briefly introduction

st.markdown("#### Introduction")
st.markdown("<div class='justified-text'>Natural language proccess technique has become an everincreasing popular topic and field of research in recent years, it can help relief human effort on multiple missions like machine translation, document classification, social media analysis and voice-to-text transition.</div>", unsafe_allow_html=True)
st.markdown(' ')
st.markdown("<div class='justified-text'>In this project research our group has focused on four typical NLP techniques, by analyzing their characteristics and performance on Sentiment analysis, we can have a more thorough understanding over these NLP techniques.</div>", unsafe_allow_html=True)
st.markdown(' ')
st.markdown("<div class='justified-text'>The four methods we adopted are: Traditional (SVM), LSTM, Transformer, and Llama. The results are evaluated based on Accuracy, Precision, Recall, and F1-Score across three datasets: IMDB, Twitter, and Google Play (sourced from Kaggle).</div>", unsafe_allow_html=True)
st.markdown(' ')


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
    if st.button("SVM"):
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
    st.write("### Transformer Introduction")
    st.image('images/Transformer/Introduction/1.png')

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
    st.markdown("<div class='justified-text'>1. The Transformer model is a neural network architecture based on the attention mechanism, first introduced by Vaswani et al. in 2017 for natural language processing (NLP) tasks. Unlike traditional recurrent neural networks (RNNs), the Transformer completely eliminates recursion and relies on self-attention to capture dependencies within sequence data, making it more efficient and flexible when handling long sequences.</div>", unsafe_allow_html=True)
    st.markdown(' ')
    st.markdown("<div class='justified-text'>2. The Transformer consists of an encoder and a decoder. The encoder maps the input sequence into high-dimensional vector representations, with each layer comprising a self-attention layer and a feedforward neural network. By stacking multiple layers, the encoder captures complex feature relationships. The decoder has a similar structure and uses the encoder's output to generate the target sequence. To recognize positional information within the sequence, the Transformer also introduces positional encoding to compensate for the unordered nature of self-attention.</div>", unsafe_allow_html=True)
    st.markdown(' ')
    st.markdown("<div class='justified-text'>3. Self-attention is the core of the Transformer, weighting each element in the input sequence by its relevance to other elements, thus obtaining global information without requiring order. This mechanism allows the Transformer to process sequences in parallel, significantly increasing training speed.</div>", unsafe_allow_html=True)
    st.markdown(' ')
    st.markdown("<div class='justified-text'>In this sentiment analysis task, due to the specific nature of the task, we chose to use only the encoder layer of the Transformer structure. This not only improved the model’s accuracy but also reduced computational costs.</div>", unsafe_allow_html=True)
    st.markdown(' ')

    if st.button("Data Preprocessing"):
        toggle_state('transformer_data_preprocessing_expanded')
    if st.session_state['transformer_data_preprocessing_expanded']:
        st.image('images/Transformer/Data Preprocessing/1.png')
        st.markdown("Create a Dataset class for each dataset, using DistilBertTokenizer to process the data to obtain input_ids, attention_mask, and labels.")
        
    if st.button("Model"):
        toggle_state('transformer_model_expanded')
    if st.session_state['transformer_model_expanded']:
        st.image('images/Transformer/Model/1.png')
        st.image('images/Transformer/Model/2.png')
        st.image('images/Transformer/Model/3.png')
        st.image('images/Transformer/Model/4.png')
        st.image('images/Transformer/Model/5.png')
        st.image('images/Transformer/Model/6.png')
    
    if st.button("Train & Test"):
        toggle_state('transformer_train_test_expanded')
    if st.session_state['transformer_train_test_expanded']:
        st.markdown('- Model: Transformer')
        st.markdown('- Optimizer:Adam(lr=5e-5)')
        st.markdown('- Loss:CrossEntropyLoss()')
        st.image('images/Transformer/Train and Test/train.png')
        st.image('images/Transformer/Train and Test/test.png')                
    
    if st.button('Results'):
        toggle_state('transformer_results_expanded')
    if st.session_state['transformer_results_expanded']:
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
                <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 86%</td>
                <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 88%</td>
                <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 84%</td>
                <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 86%</td>
            </tr>
            <tr>
                <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> Twitter</td>
                <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 77%</td>
                <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 82%</td>
                <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 69%</td>
                <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 75%</td>
            </tr>
            <tr>
                <td style='border: 0.5px solid black; padding: 10px; width: 250px;'> GooglePlay</td>
                <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 81%</td>
                <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 82%</td>
                <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 89%</td>
                <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 85%</td>
            </tr>
            </table>
            """, unsafe_allow_html=True)  
        st.markdown("<div class='justified-text'>Overall, SVM is well-suited for both classification and regression tasks, particularly when dealing with high-dimensional data but limited sample sizes. In this case, the average runtime for the three datasets is approximately 1 hour. However, traditional machine learning algorithms tend to perform slower on larger datasets compared to deep learning models. Despite this, traditional machine learning models are simpler and more interpretable because they lack the complex structures and numerous layers found in deep neural networks.</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='justified-text'>1. IMDB</div>", unsafe_allow_html=True)
        st.markdown("<div class='justified-text'>Accuracy: 0.8612 - IMDB has the highest accuracy rate among the three platforms.</div>", unsafe_allow_html=True)
        st.markdown("<div class='justified-text'>Recall: 0.842 - IMDB also has a high recall rate.</div>", unsafe_allow_html=True)
        st.markdown("<div class='justified-text'>F1-score: 0.8601 - IMDB has the highest F1-score, indicating that its model performs best in balancing precision and recall.</div>", unsafe_allow_html=True)
        st.markdown(' ')     

        st.markdown("<div class='justified-text'>2. Twitter:</div>", unsafe_allow_html=True)
        st.markdown("<div class='justified-text'>Accuracy: 0.7728 - Twitter has the lowest accuracy rate, which may require further model optimization.</div>", unsafe_allow_html=True)
        st.markdown("<div class='justified-text'>Recall: 0.6895 - Twitter's recall rate is significantly lower than IMDB and Google Play.</div>", unsafe_allow_html=True)
        st.markdown("<div class='justified-text'>F1-score: 0.747 - Twitter's F1-score is also the lowest among the three. This may indicate that Twitter's model struggles more in balancing precision and recall compared to IMDB and Google Play.</div>", unsafe_allow_html=True)
        st.markdown(' ')

        st.markdown("<div class='justified-text'>3. Google Play:</div>", unsafe_allow_html=True)  
        st.markdown("<div class='justified-text'>Accuracy: 0.8192 - Google Play's accuracy is lower than IMDB's, but still relatively high.</div>", unsafe_allow_html=True)  
        st.markdown("<div class='justified-text'>Recall: 0.8081 - Google Play's recall rate is slightly lower than IMDB's.</div>", unsafe_allow_html=True)  
        st.markdown("<div class='justified-text'>F1-score: 0.8514 - Google Play's F1-score is also very high, suggesting its model performs well.</div>", unsafe_allow_html=True)  
        st.markdown(' ')  

        st.markdown("<div class='justified-text'>Overall, IMDB's model performance is the strongest, followed by Google Play, while Twitter's model is relatively weaker. For Twitter, further model optimization could be considered to improve its accuracy and recall.</div>", unsafe_allow_html=True)
        st.markdown(' ')

        st.markdown('**Training thoughts**')
        st.markdown("<div class='justified-text'>Q1:Why does increasing the number of Transformer block layers lead to a decline in model performance?</div>", unsafe_allow_html=True)
        st.markdown("<div class='justified-text'>A1:Consider the overfitting .A deeper model might lead to overfitting during training, especially when the training data is limited.</div>", unsafe_allow_html=True)
        st.markdown(' ')
        st.markdown("<div class='justified-text'>Q2: Why learning rate has a significant impact on model performance?</div>", unsafe_allow_html=True)
        st.markdown("<div class='justified-text'>A2: Learning rate too high: If the learning rate is too large, the steps taken during each update are too big, and the model might 'jump' around the minimum of the loss function rather than converging to it.</div>", unsafe_allow_html=True)
        st.markdown("<div class='justified-text'>Learning rate too low: A low learning rate may cause the model to get stuck in a local minimum of the loss function without being able to escape to a better solution.</div>", unsafe_allow_html=True)
        st.markdown(' ')
       

# Display Traditional content if expanded
if st.session_state['traditional_expanded']:
    st.write("### Traditional Introduction")
    st.image('images/SVM/traditional machine learning.jpg')

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
    st.markdown("<div class='justified-text'>Traditional machine learning refers to a set of algorithms that were widely used before the advent of deep learning. These methods rely on statistical and mathematical models to extract relevant features from raw data and process them to generate predictions or insights. Common examples of traditional machine learning algorithms include Linear Regression, Support Vector Machines (SVM), and K-Nearest Neighbors (K-NN), among others.</div>", unsafe_allow_html=True)
    st.markdown(' ')
    st.markdown("<div class='justified-text'>To evaluate the performance of different algorithms, this analysis employs SVM as the traditional machine learning model. The core principle of SVM is to identify a hyperplane that maximizes the margin between data points from different classes, thereby effectively separating them into distinct categories.</div>", unsafe_allow_html=True)
    
    if st.button('Data Preprocessing'):
        toggle_state('svm_data_preprocessing_expanded')
    if st.session_state['svm_data_preprocessing_expanded']:
        st.image('images/SVM/Data Preprocessing.jpg')
        st.markdown("<div class='justified-text'>During data preprocessing, entries with sentiment labels such as Neutral were removed, retaining only those labeled as Negative and Positive.</div>", unsafe_allow_html=True)

    if st.button('Text Processing'):
        toggle_state('svm_text_processing_expanded')
    if st.session_state['svm_text_processing_expanded']:
        st.image('images/SVM/TF-IDF.jpg')
        st.markdown("<div class='justified-text'>The Word Frequency - Inverse Document Frequency(TF-IDF) Vectorization has been used over here. The text data is transformed into a TF-IDF feature matrix. After several iterations, only the top 5,000 features are selected to balance the feature dimensionality and avoid issues caused by an excessively high or low number of features. Finally, the sentiment labels for the training and test datasets are stored in y_train and y_test, respectively.</div>", unsafe_allow_html=True)

    if st.button('Training Model'):
        toggle_state('svm_training_model_expanded')
    if st.session_state['svm_training_model_expanded']:
        st.image('images/SVM/SVM Model.jpg')
        st.markdown("<div class='justified-text'>A Support Vector Machine (SVM) model with a linear kernel is constructed, and the training data along with their corresponding labels are used to fit the model.</div>", unsafe_allow_html=True)
    
    if st.button('Results'):
        toggle_state('svm_results_expanded')
    if st.session_state['svm_results_expanded']:
        st.image('images/SVM/Results.jpg')
        st.markdown("<div class='justified-text'>The trained model is used to make predictions on the test data, returning the predicted sentiment labels. The accuracy of the predictions is calculated as the percentage of correctly classified samples out of the total number of samples. Additionally, evaluation metrics such as Precision, Recall, and F1 score are also reported.</div>", unsafe_allow_html=True)

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
                <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 89%</td>
                <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 89%</td>
                <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 89%</td>
                <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 89%</td>
            </tr>
            <tr>
                <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> Twitter</td>
                <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 79%</td>
                <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 79%</td>
                <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 79%</td>
                <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 79%</td>
            </tr>
            <tr>
                <td style='border: 0.5px solid black; padding: 10px; width: 250px;'> GooglePlay</td>
                <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 80%</td>
                <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 80%</td>
                <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 80%</td>
                <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 80%</td>
            </tr>
            </table>
            """, unsafe_allow_html=True)  
        st.markdown("<div class='justified-text'>Overall, SVM is well-suited for both classification and regression tasks, particularly when dealing with high-dimensional data but limited sample sizes. In this case, the average runtime for the three datasets is approximately 1 hour. However, traditional machine learning algorithms tend to perform slower on larger datasets compared to deep learning models. Despite this, traditional machine learning models are simpler and more interpretable because they lack the complex structures and numerous layers found in deep neural networks.</div>", unsafe_allow_html=True)





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
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'>SVM</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'>LSTM</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'>Transformer</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'>Llama3.2</td>
        </tr>
        <tr>
            <td style='border: 0.5px solid black; padding: 10px; width: 200px;'> Accuarcy on IMDB</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 89%</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 87%</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 86%</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 95%</td>
        </tr>
        <tr>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> Accuarcy on Twitter</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 79%</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 72%</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 77%</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 56%</td>
        </tr>
        <tr>
            <td style='border: 0.5px solid black; padding: 10px; width: 250px;'> Accuarcy on GooglePlay</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 80%</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 75%</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 82%</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 74%</td>
        </tr>
        </table>
        """, unsafe_allow_html=True)

# precision
if st.session_state['precision_expanded']:  
    st.markdown("""
        <table style='width:100%; border: 0.5px solid black; border-collapse: collapse;'>
        <tr>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> </td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'>SVM</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'>LSTM</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'>Transformer</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'>Llama3.2</td>
        </tr>
        <tr>
            <td style='border: 0.5px solid black; padding: 10px; width: 200px;'> Precision on IMDB</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 89%</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 84%</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 88%</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 95%</td>
        </tr>
        <tr>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> Precision on Twitter</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 79%</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 71%</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 82%</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 54%</td>
        </tr>
        <tr>
            <td style='border: 0.5px solid black; padding: 10px; width: 250px;'> Precision on GooglePlay</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 80%</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 82%</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 82%</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 77%</td>
        </tr>
        </table>
        """, unsafe_allow_html=True)

# recall
if st.session_state['recall_expanded']:  
    st.markdown("""
        <table style='width:100%; border: 0.5px solid black; border-collapse: collapse;'>
        <tr>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> </td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'>SVM</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'>LSTM</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'>Transformer</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'>Llama3.2</td>
        </tr>
        <tr>
            <td style='border: 0.5px solid black; padding: 10px; width: 200px;'> Recall on IMDB</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 89%</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 90%</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 84%</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 95%</td>
        </tr>
        <tr>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> Recall on Twitter</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 79%</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 71%</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 69%</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 62%</td>
        </tr>
        <tr>
            <td style='border: 0.5px solid black; padding: 10px; width: 250px;'> Recall on GooglePlay</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 80%</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 75%</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 89%</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 84%</td>
        </tr>
        </table>
        """, unsafe_allow_html=True)

# f1-score
if st.session_state['f1_expanded']:  
    st.markdown("""
        <table style='width:100%; border: 0.5px solid black; border-collapse: collapse;'>
        <tr>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> </td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'>SVM</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'>LSTM</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'>Transformer</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'>Llama3.2</td>
        </tr>
        <tr>
            <td style='border: 0.5px solid black; padding: 10px; width: 200px;'> F1-score on IMDB</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 89%</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 87%</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 86%</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 95%</td>
        </tr>
        <tr>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> F1-score on Twitter</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 79%</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 71%</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 75%</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 58%</td>
        </tr>
        <tr>
            <td style='border: 0.5px solid black; padding: 10px; width: 250px;'> F1-score on GooglePlay</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 80%</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 78%</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 85%</td>
            <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> 80%</td>
        </tr>
        </table>
        """, unsafe_allow_html=True)