import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import '../models/soil_reading.dart';

class SoilFirestoreService {
  final _db = FirebaseFirestore.instance;
  final _auth = FirebaseAuth.instance;

  DocumentReference<Map<String, dynamic>>? _latestRef() {
    final user = _auth.currentUser;
    if (user == null) return null;

    return _db
        .collection("users")
        .doc(user.uid)
        .collection("soil_readings")
        .doc("latest");
  }

  Stream<SoilReading?> latestSoilStream() {
    final ref = _latestRef();
    if (ref == null) return const Stream.empty();

    return ref.snapshots().map((doc) {
      if (!doc.exists) return null;
      return SoilReading.fromDoc(doc);
    });
  }

  Future<SoilReading?> fetchLatestSoilReading() async {
    final ref = _latestRef();
    if (ref == null) return null;

    final doc = await ref.get();
    if (!doc.exists) return null;
    return SoilReading.fromDoc(doc);
  }

  Future<void> saveLatestSoilInputs({
    required double temperature,
    required double ph,
    required double n,
    required double p,
    required double k,
  }) async {
    final ref = _latestRef();
    if (ref == null) {
      throw Exception("Not logged in");
    }

    await ref.set({
      "temperature": temperature,
      "ph": ph,
      "N": n,
      "P": p,
      "K": k,
      "updatedAt": FieldValue.serverTimestamp(),
    }, SetOptions(merge: true));
  }
}
