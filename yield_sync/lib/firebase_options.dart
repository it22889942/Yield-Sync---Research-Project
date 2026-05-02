import 'package:firebase_core/firebase_core.dart' show FirebaseOptions;
import 'package:flutter/foundation.dart' show kIsWeb;

class DefaultFirebaseOptions {
  static FirebaseOptions get currentPlatform {
    if (kIsWeb) {
      return const FirebaseOptions(
        apiKey: "AIzaSyAr6Ywt6DIBONS0y7qyvoPd3fIOu17ESPg",
        appId: "1:899385662271:android:003a319c3c69b2afc0518a",
        messagingSenderId: "899385662271",
        projectId: "yield-sync-b1f09",
        authDomain: "yield-sync-b1f09.firebaseapp.com",
        storageBucket: "yield-sync-b1f09.firebasestorage.app",
      );
    }

    // Android will use google-services.json automatically.
    // But Firebase still allows options null on Android.
    return const FirebaseOptions(
      apiKey: "DUMMY",
      appId: "DUMMY",
      messagingSenderId: "DUMMY",
      projectId: "DUMMY",
    );
  }
}
