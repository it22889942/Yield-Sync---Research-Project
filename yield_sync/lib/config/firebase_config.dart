import 'package:firebase_core/firebase_core.dart';
import 'package:flutter/foundation.dart' show kIsWeb;
import '../firebase_options.dart';

class FirebaseConfig {
  FirebaseConfig._();

  static const bool enabled = true;

  static Future<void> init() async {
    if (!enabled) return;

    if (kIsWeb) {
      await Firebase.initializeApp(
        options: DefaultFirebaseOptions.currentPlatform,
      );
    } else {
      await Firebase.initializeApp(); // Android (google-services.json)
    }
  }
}
