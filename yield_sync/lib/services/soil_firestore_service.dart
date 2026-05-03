import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import '../models/soil_reading.dart';

class SoilFirestoreService {
  final _db = FirebaseFirestore.instance;
  final _auth = FirebaseAuth.instance;

  Stream<SoilReading?> latestSoilStream() {
    final user = _auth.currentUser;
    if (user == null) return const Stream.empty();

    final ref = _db
        .collection("users")
        .doc(user.uid)
        .collection("soil_readings")
        .doc("latest");

    return ref.snapshots().map((doc) {
      if (!doc.exists) return null;
      return SoilReading.fromDoc(doc);
    });
  }
}
