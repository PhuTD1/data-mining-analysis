// src/components/useinterface.js
import React, { useState } from 'react';
import './ChatPage.css';
// import { BlockMath } from 'react-katex';
const ChatPage = () => {
  const [messages, setMessages] = useState([]);
  const [newMessage, setNewMessage] = useState('');

  // request
  const sendDatatoBackend = async (message) => {
    try {
      console.log(message)
      const response = await fetch('http://127.0.0.1:5000/server', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ key: 'successfully!!!', message }),
      });
      const result = await response.json();   // response from server
      setMessages(prevMessages => [...prevMessages, { text: result.message, sender: 'bot' }]);
    } catch (error) {
      console.log('Error:', error);
    }
  };

  const handleSendMessage = () => {
    if (newMessage.trim() !== '') {
      setMessages([...messages, { text: newMessage, sender: 'user' }]);
      sendDatatoBackend(newMessage)
      setNewMessage('');
    }
  };



  return (
    <div className="ChatApp">
      <div className="message-container">
        {messages.map((message, index) => (
          <div key={index} className={`message ${message.sender === 'user' ? 'user-message' : 'bot-message'}`}>
            <div className={`message-sender ${message.sender === 'user' ? 'user-sender' : 'bot-sender'}`}>
              {message.sender === 'user' ? 'You ' : 'AI-detect'}
            </div>
            <div className={`message-text ${message.sender === 'user' ? 'user-text' : 'bot-text'}`}>{message.text}</div>
          </div>
        ))}
      </div>
      <div className="input-container">
        <input
          type="text"
          placeholder="Type a message..."
          value={newMessage}
          onChange={(e) => setNewMessage(e.target.value)}
        />
        <button onClick={handleSendMessage}><span role="img" aria-label="Send">
          ✉️
        </span></button>
      </div>
    </div>
  );
};

export default ChatPage;
