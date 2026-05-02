import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';

class ForecastDB {
  static final _db = FirebaseFirestore.instance;

  static String? get uid => FirebaseAuth.instance.currentUser?.uid;

  static Future<String> saveForecast({
    required Map<String, dynamic> input,
    required Map<String, dynamic> predict,
    required Map<String, dynamic> recommendation,
    required Map<String, dynamic> history,
  }) async {
    final u = uid;
    if (u == null) throw Exception("User not logged in");

    final ref = await _db
        .collection("users")
        .doc(u)
        .collection("market_forecasts")
        .add({
      "createdAt": FieldValue.serverTimestamp(),
      "input": input,
      "predict": predict,
      "recommendation": recommendation,
      "history": history,
    });

    return ref.id;
  }
}
