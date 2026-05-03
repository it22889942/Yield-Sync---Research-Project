import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';

class ReviewService {
  static final _db = FirebaseFirestore.instance;
  static final _auth = FirebaseAuth.instance;

  static const String reviewsCol = "reviews";

  static Future<void> addReview({
    required String bookingId,
    required String itemType, // "labour" | "equipment"
    required String itemId,
    required String itemName,
    required String ownerName,
    required String sentiment, // "good" | "bad"
    required double rating, // 1..5
    required String comment,
  }) async {
    final u = _auth.currentUser;
    if (u == null) throw Exception("Not logged in");

    final userDoc = await _db.collection("users").doc(u.uid).get();
    final user = userDoc.data() ?? {};
    final reviewByName =
        (user["username"] ?? user["firstName"] ?? user["name"] ?? "User")
            .toString();
    final reviewByPhone = (user["phone"] ?? "—").toString();
    final reviewByEmail = (user["email"] ?? "—").toString();

    await _db.collection(reviewsCol).add({
      "reviewByUid": u.uid,
      "reviewByName": reviewByName,
      "reviewByPhone": reviewByPhone,
      "reviewByEmail": reviewByEmail,
      "bookingId": bookingId,
      "itemType": itemType,
      "itemId": itemId,
      "itemName": itemName,
      "ownerName": ownerName,
      "sentiment": sentiment,
      "rating": rating,
      "comment": comment.trim(),
      "createdAt": FieldValue.serverTimestamp(),
    });
  }

  // ✅ NO INDEX REQUIRED (no orderBy)
  static Stream<QuerySnapshot<Map<String, dynamic>>> labourReviews(
      String labourId) {
    return _db
        .collection(reviewsCol)
        .where("itemType", isEqualTo: "labour")
        .where("itemId", isEqualTo: labourId)
        .snapshots();
  }

  // ✅ NO INDEX REQUIRED (no orderBy)
  static Stream<QuerySnapshot<Map<String, dynamic>>> equipmentReviews(
      String equipmentId) {
    return _db
        .collection(reviewsCol)
        .where("itemType", isEqualTo: "equipment")
        .where("itemId", isEqualTo: equipmentId)
        .snapshots();
  }
}
