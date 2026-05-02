import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';

class PredictionHistoryService {
  final _db = FirebaseFirestore.instance;
  final _auth = FirebaseAuth.instance;

  Future<void> saveCropPrediction({
    required Map<String, dynamic> request,
    required Map<String, dynamic> response,
    required double lat,
    required double lon,
    required String place,
  }) async {
    final user = _auth.currentUser;
    if (user == null) throw Exception("Not logged in");

    final uid = user.uid;

    await _db.collection("users").doc(uid).collection("crop_predictions").add({
      "uid": uid,
      "place": place,
      "lat": lat,
      "lon": lon,
      "request": request,
      "response": response,
      "createdAt": FieldValue.serverTimestamp(),
    });
  }
}
