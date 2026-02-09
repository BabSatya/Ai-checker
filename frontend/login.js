import { auth } from "./firebase.js";
import { signInWithEmailAndPassword } from "firebase/auth";

async function login(email, password) {
  try {
    const userCred = await signInWithEmailAndPassword(
      auth,
      email,
      password
    );

    alert("Login Success!");
    window.location.href = "dashboard.html";

  } catch (err) {
    alert(err.message);
  }
}
