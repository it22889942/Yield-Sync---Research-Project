import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';

class AuthService {
  final FirebaseAuth _auth = FirebaseAuth.instance;
  final FirebaseFirestore _db = FirebaseFirestore.instance;

  // ✅ Change this if your Firestore collection name is different
  static const String EQUIPMENT_COLLECTION = "equipments";

  // ----------------------------
  // Existing: Farmer/Seller basic register (KEEP to avoid breaking other files)
  // ----------------------------
  Future<void> registerUser({
    required String firstName,
    required String lastName,
    required String username,
    required String phone,
    required String email,
    required String password,
    required String userType,
  }) async {
    // ✅ Keep old behavior (but you can still use farmer)
    final type = userType.trim().toLowerCase();
    if (type != "farmer" && type != "seller") {
      throw ArgumentError("Invalid userType. Only farmer/seller allowed.");
    }

    final cred = await _auth.createUserWithEmailAndPassword(
      email: email.trim(),
      password: password,
    );

    final uid = cred.user!.uid;

    await _db.collection("users").doc(uid).set({
      "uid": uid,
      "firstName": firstName.trim(),
      "lastName": lastName.trim(),
      "username": username.trim(),
      "phone": phone.trim(),
      "email": email.trim(),
      "userType": type,
      "isActive": true,
      "createdAt": FieldValue.serverTimestamp(),
    });
  }

  // ----------------------------
  // ✅ Labour register with Labour_ID check
  // Firestore: labours/{Labour_ID}
  // ----------------------------
  Future<void> registerLabour({
    required String labourId, // ex: L0001
    required String firstName,
    required String lastName,
    required String username,
    required String phone,
    required String email,
    required String password,
  }) async {
    final id = labourId.trim();

    // 1) verify labour exists in dataset
    final snap = await _db.collection("labours").doc(id).get();
    if (!snap.exists) {
      throw Exception("Invalid Labour ID ($id). Not found in dataset.");
    }

    // 2) create auth user
    final cred = await _auth.createUserWithEmailAndPassword(
      email: email.trim(),
      password: password,
    );

    final uid = cred.user!.uid;

    // 3) create users doc mapping to labourId
    await _db.collection("users").doc(uid).set({
      "uid": uid,
      "firstName": firstName.trim(),
      "lastName": lastName.trim(),
      "username": username.trim(),
      "phone": phone.trim(),
      "email": email.trim(),
      "userType": "labour",
      "labourId": id,
      "isActive": true,
      "createdAt": FieldValue.serverTimestamp(),
    });
  }

  // ----------------------------
  // ✅ NEW: Equipment register with Equipment_ID check
  // Firestore: equipments/{Equipment_ID}
  // ----------------------------
  Future<void> registerEquipment({
    required String equipmentId, // ex: E0001 or whatever your dataset uses
    required String firstName,
    required String lastName,
    required String username,
    required String phone,
    required String email,
    required String password,
  }) async {
    final id = equipmentId.trim();

    // 1) verify equipment exists in dataset
    final snap = await _db.collection(EQUIPMENT_COLLECTION).doc(id).get();
    if (!snap.exists) {
      throw Exception("Invalid Equipment ID ($id). Not found in dataset.");
    }

    // 2) create auth user
    final cred = await _auth.createUserWithEmailAndPassword(
      email: email.trim(),
      password: password,
    );

    final uid = cred.user!.uid;

    // 3) create users doc mapping to equipmentId
    await _db.collection("users").doc(uid).set({
      "uid": uid,
      "firstName": firstName.trim(),
      "lastName": lastName.trim(),
      "username": username.trim(),
      "phone": phone.trim(),
      "email": email.trim(),
      "userType": "equipment", // ✅ IMPORTANT
      "equipmentId": id, // ✅ IMPORTANT
      "isActive": true,
      "createdAt": FieldValue.serverTimestamp(),
    });
  }

  // ----------------------------
  // Existing: Seller register (KEEP to avoid breaking other files)
  // (You can delete later if not used anywhere)
  // ----------------------------
  Future<void> registerSeller({
    required String sellerId,
    required String firstName,
    required String lastName,
    required String username,
    required String phone,
    required String email,
    required String password,
  }) async {
    final id = sellerId.trim();

    final snap = await _db.collection("sellers").doc(id).get();
    if (!snap.exists) {
      throw Exception("Invalid Seller ID ($id). Not found in dataset.");
    }

    final cred = await _auth.createUserWithEmailAndPassword(
      email: email.trim(),
      password: password,
    );

    final uid = cred.user!.uid;

    await _db.collection("users").doc(uid).set({
      "uid": uid,
      "firstName": firstName.trim(),
      "lastName": lastName.trim(),
      "username": username.trim(),
      "phone": phone.trim(),
      "email": email.trim(),
      "userType": "seller",
      "sellerId": id,
      "isActive": true,
      "createdAt": FieldValue.serverTimestamp(),
    });
  }

  // ----------------------------
  // Login
  // ----------------------------
  Future<void> login({
    required String email,
    required String password,
  }) async {
    await _auth.signInWithEmailAndPassword(
      email: email.trim(),
      password: password,
    );
  }

  Future<String> getCurrentUserType() async {
    final user = _auth.currentUser;
    if (user == null) return "unknown";

    final snap = await _db.collection("users").doc(user.uid).get();
    final data = snap.data();
    return (data?["userType"] ?? "unknown").toString().toLowerCase();
  }

  Future<String?> getMyLabourId() async {
    final user = _auth.currentUser;
    if (user == null) return null;

    final snap = await _db.collection("users").doc(user.uid).get();
    return snap.data()?["labourId"]?.toString();
  }

  // ✅ NEW
  Future<String?> getMyEquipmentId() async {
    final user = _auth.currentUser;
    if (user == null) return null;

    final snap = await _db.collection("users").doc(user.uid).get();
    return snap.data()?["equipmentId"]?.toString();
  }

  Future<String?> getMySellerId() async {
    final user = _auth.currentUser;
    if (user == null) return null;

    final snap = await _db.collection("users").doc(user.uid).get();
    return snap.data()?["sellerId"]?.toString();
  }

  Future<void> logout() async => _auth.signOut();

  User? get currentUser => _auth.currentUser;

  Future<Map<String, dynamic>?> getMyProfile() async {
    final user = _auth.currentUser;
    if (user == null) return null;

    final snap = await _db.collection("users").doc(user.uid).get();
    return snap.data();
  }

  Future<void> updateMyProfile({
    required String firstName,
    required String lastName,
    required String username,
    required String phone,
  }) async {
    final user = _auth.currentUser;
    if (user == null) throw Exception("Not logged in");

    await _db.collection("users").doc(user.uid).update({
      "firstName": firstName.trim(),
      "lastName": lastName.trim(),
      "username": username.trim(),
      "phone": phone.trim(),
      "updatedAt": FieldValue.serverTimestamp(),
    });
  }
}
