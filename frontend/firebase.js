import { initializeApp } from "firebase/app";
import { getFirestore } from "firebase/firestore";

const firebaseConfig = {
  apiKey: "AIzaSyA3HEF5WtuFPxkc3EBmhkTef-fQvk4td_I",
  authDomain: "journal-35923.firebaseapp.com",
  projectId: "journal-35923",
  storageBucket: "journal-35923.firebasestorage.app",
  messagingSenderId: "140616825850",
  appId: "1:140616825850:web:27d47a5b9b083ac6968243",
};

const app = initializeApp(firebaseConfig);

export const db = getFirestore(app);

console.log("🔥 Firebase Initialized Successfully");
