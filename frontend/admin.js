import { auth, db } from "./firebase.js";
import {
  createUserWithEmailAndPassword
} from "firebase/auth";

import { doc, setDoc } from "firebase/firestore";

async function createUser(email, password, tokens) {
  try {

    // Create Firebase Auth Account
    const userCredential = await createUserWithEmailAndPassword(
      auth,
      email,
      password
    );

    const uid = userCredential.user.uid;

    // Store Extra Info in Firestore
    await setDoc(doc(db, "users", uid), {
      email: email,
      role: "user",
      tokens: tokens,
      createdAt: new Date()
    });

    alert("User Created Successfully!");

  } catch (error) {
    alert("Error: " + error.message);
  }
}
