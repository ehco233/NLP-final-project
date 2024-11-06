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
if 'data_preprocessing_expanded' not in st.session_state:
    st.session_state['data_preprocessing_expanded'] = False
if 'word_embedding_expanded' not in st.session_state:
    st.session_state['word_embedding_expanded'] = False
if 'model_training_expanded' not in st.session_state:
    st.session_state['model_training_expanded'] = False
if 'results_expanded' not in st.session_state:
    st.session_state['results_expanded'] = False


def toggle_state(state_name):
    st.session_state[state_name] = not st.session_state[state_name]

st.title("NLP Final Project A13")
st.markdown("Sentiment Analysis")

# table
st.markdown("#### Final Results")
st.markdown("""
    <table style='width:100%; border: 2px solid black; border-collapse: collapse;'>
    <tr>
        <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> </td>
        <td style='border: 0.5px solid black; padding: 10px; width: 150px;'>Traditional</td>
        <td style='border: 0.5px solid black; padding: 10px; width: 150px;'>LSTM</td>
        <td style='border: 0.5px solid black; padding: 10px; width: 150px;'>Transformer</td>
        <td style='border: 0.5px solid black; padding: 10px; width: 150px;'>llma3.2</td>
    </tr>
    <tr>
        <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> Accuarcy on IMDB</td>
        <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> </td>
        <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> </td>
        <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> </td>
        <td style='border: 0.5px solid black; padding: 10px; width: 150px;'> </td>
    </tr>
    </table>
    """, unsafe_allow_html=True)

st.markdown("<hr style='border: 0.5px dashed black;'>", unsafe_allow_html=True)

# LSTM Button
if st.button("LSTM"):
    toggle_state('lstm_expanded')
if st.session_state['lstm_expanded']:
    st.write("##### LSTM Options")
    st.write("- Long Short-Term Memory(LSTM) is a type of recurrent neural network(RNN) used for sequence prediction problems.")
    st.write('- It is capable of learning long-term dependencies and is effective for time-series data')
    
    lstm_col1, lstm_col2 = st.columns([1, 16])
    with lstm_col2:
        if st.button("Data Preprocessing"):
            toggle_state('data_preprocessing_expanded')
        if st.session_state['data_preprocessing_expanded']:
            st.markdown("- Remove special characters")
            st.image('/Users/caro/Desktop/data_preprocessing.png')
            st.markdown("- Stopwords Removal")
            st.markdown("- lemmatization")

        if st.button("Word Embedding"):
            toggle_state('word_embedding_expanded')
        if st.session_state['word_embedding_expanded']:
            st.markdown("##### Word2Vec")
            st.markdown("- Mode: Skip-gram")
            st.image('/Users/caro/Desktop/word_embedding.png')

        if st.button("Model Training"):
            toggle_state('model_training_expanded')
        if st.session_state['model_training_expanded']:
            st.markdown("##### Model: Bi-Directional LSTM")
            st.markdown('- Bi-Directional RNNs consist of two separate RNNs: one moving forward through the input sequence and the other moving backward.')
            st.image('/Users/caro/Desktop/model_training.png')
            st.markdown('- Loss Function: BinaryCrossentropy')
            st.markdown('- Metrics: accuracy')
            st.markdown('- Callback: EarlyStopping, ReduceLOnPlateau')
            st.image('/Users/caro/Desktop/callback.png')
            st.markdown('- Model fit parameters: epochs, batch_size')
            st.image('/Users/caro/Desktop/modelfit.png')
        
        if st.button("Results"):
            toggle_state('results_expanded')
        if st.session_state['results_expanded']:
            st.markdown('##### Results on IMDB dataset')
            st.markdown('- val_accuracy: <span style="color: red;">**0.8647**</span>', unsafe_allow_html=True)
            st.markdown('- Early stopping occurred at the 8th epoch, with the best result achieved at the <span style="color: red;">**5th**</span> epoch.', unsafe_allow_html=True)
            st.markdown('- ReduceLROnPlateau reducing learning rate to 4.999999873689376e-05 at 7th epoch')
            st.image('/Users/caro/Desktop/output.png')
            st.image('/Users/caro/Desktop/results.png')
            

st.markdown("<hr style='border: 0.5px dashed green;'>", unsafe_allow_html=True)

# Transformer Button
if st.button("Transformer"):
    toggle_state('transformer_expanded')
if st.session_state['transformer_expanded']:
    st.write("### Transformer Options")

st.markdown("<hr style='border: 0.5px dashed orange;'>", unsafe_allow_html=True)

# Traditional Button
if st.button("Traditional"):
    toggle_state('traditional_expanded')
if st.session_state['traditional_expanded']:
    st.write("### Traditional Options")

st.markdown("<hr style='border: 0.5px dashed blue;'>", unsafe_allow_html=True)

# LLMA Button
if st.button("LLMA"):
    toggle_state('llma_expanded')
if st.session_state['llma_expanded']:
    st.write("### LLMA Options")

st.markdown("<hr style='border: 0.5px dashed yellow;'>", unsafe_allow_html=True)
