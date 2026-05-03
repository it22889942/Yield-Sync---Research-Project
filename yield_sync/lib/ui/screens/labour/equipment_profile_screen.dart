import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';

import '../../../utils/app_colors.dart';
import '../../../services/auth_service.dart';
import '../../../services/equipment_api.dart';
import '../../../services/equipment_booking_service.dart';
import '../../../services/review_service.dart';

const String EQUIPMENT_COLLECTION = "equipments";
const String EQUIPMENT_BOOKINGS_COLLECTION = "equipment_bookings";

class EquipmentProfileScreen extends StatefulWidget {
  const EquipmentProfileScreen({super.key});

  @override
  State<EquipmentProfileScreen> createState() => _EquipmentProfileScreenState();
}

class _EquipmentProfileScreenState extends State<EquipmentProfileScreen> {
  bool _loading = true;
  String? _error;

  Map<String, dynamic>? _equipment;
  String _equipmentId = "";

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    setState(() {
      _loading = true;
      _error = null;
    });

    try {
      final equipmentId = await AuthService().getMyEquipmentId();
      if (equipmentId == null || equipmentId.trim().isEmpty) {
        throw Exception("Equipment ID not linked to this user.");
      }

      final eid = equipmentId.trim();
      _equipmentId = eid;

      // Try backend first
      try {
        final details = await EquipmentApi.getItem(eid);
        if (!mounted) return;

        setState(() {
          _equipment = {
            "Equipment_ID": details.id,
            "Equipment_Type": details.equipmentType,
            "For_Crop": details.forCrop,
            "Hourly_Rate_LKR": details.hourlyRate,
            "Daily_Rate_LKR": details.dailyRate,
            "Main_District": details.mainDistrict,
            "Nearest_Major_District": details.nearestMajorDistrict,
            "Condition": details.condition,
            "Equipment_Owner_Name": details.ownerName,
            "Owner_Contact_No": details.ownerContact,
            "Available_Day": details.availableDay,
            "Available_Time": details.availableTime,
          };
        });
      } catch (_) {
        final snap = await FirebaseFirestore.instance
            .collection(EQUIPMENT_COLLECTION)
            .doc(eid)
            .get();

        if (!snap.exists) {
          throw Exception("Equipment doc missing in Firestore: $eid");
        }
        if (!mounted) return;
        setState(() => _equipment = snap.data());
      }
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  // ✅ Cleaner status palette for modern UI
  Color _statusColor(String s) {
    switch (s) {
      case "accepted":
        return const Color(0xFF15B77E);
      case "rejected":
        return const Color(0xFFE25555);
      case "cancelled":
        return const Color(0xFFF29B38);
      case "completed":
        return const Color(0xFF2BB3D1);
      default:
        return const Color(0xFF6B7C77);
    }
  }

  IconData _statusIcon(String s) {
    switch (s) {
      case "accepted":
        return Icons.check_circle_rounded;
      case "rejected":
        return Icons.cancel_rounded;
      case "cancelled":
        return Icons.do_not_disturb_on_rounded;
      case "completed":
        return Icons.verified_rounded;
      default:
        return Icons.hourglass_bottom_rounded;
    }
  }

  String _statusText(String s) {
    if (s.trim().isEmpty) return "PENDING";
    return s.toUpperCase();
  }

  Future<void> _logout() async {
    await FirebaseAuth.instance.signOut();
    if (!mounted) return;
    Navigator.of(context).pushNamedAndRemoveUntil("/login", (r) => false);
  }

  Future<void> _updateBookingStatus({
    required String bookingId,
    required String status,
  }) async {
    await FirebaseFirestore.instance
        .collection(EQUIPMENT_BOOKINGS_COLLECTION)
        .doc(bookingId)
        .update({
      "status": status,
      "updatedAt": FieldValue.serverTimestamp(),
    });
  }

  Future<void> _linkMyEquipment() async {
    try {
      await EquipmentBookingService.linkMyEquipmentOwnerUid();
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("Linked ✅ ownerUid set in /equipments")),
      );
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("Link failed: $e")),
      );
    }
  }

  // ✅ Modern card builder
  Widget _card({required Widget child, EdgeInsets? padding}) {
    return Container(
      padding: padding ?? const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(22),
        border: Border.all(color: AppColors.border),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.045),
            blurRadius: 18,
            offset: const Offset(0, 12),
          ),
        ],
      ),
      child: child,
    );
  }

  Widget _chip({
    required IconData icon,
    required String text,
    Color? bg,
    Color? fg,
    Color? border,
  }) {
    final bgC = bg ?? AppColors.primary.withOpacity(0.14);
    final fgC = fg ?? AppColors.darkGreen;
    final bd = border ?? AppColors.border;

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 7),
      decoration: BoxDecoration(
        color: bgC,
        borderRadius: BorderRadius.circular(999),
        border: Border.all(color: bd),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 16, color: fgC),
          const SizedBox(width: 6),
          Text(
            text,
            style: TextStyle(fontWeight: FontWeight.w900, color: fgC),
          ),
        ],
      ),
    );
  }

  Widget _sectionTitle(String t, {IconData? icon}) {
    return Row(
      children: [
        if (icon != null) ...[
          Container(
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: AppColors.primary.withOpacity(0.14),
              borderRadius: BorderRadius.circular(14),
              border: Border.all(color: AppColors.border),
            ),
            child: Icon(icon, size: 18, color: AppColors.darkGreen),
          ),
          const SizedBox(width: 10),
        ],
        Expanded(
          child: Text(
            t,
            style: const TextStyle(
              fontWeight: FontWeight.w900,
              fontSize: 16,
              color: AppColors.textDark,
            ),
          ),
        ),
      ],
    );
  }

  // ✅ Reviews UI widget (upgraded UI + average summary)
  Widget _reviewsSection() {
    final stream = (_equipmentId.isEmpty)
        ? const Stream<QuerySnapshot<Map<String, dynamic>>>.empty()
        : ReviewService.equipmentReviews(_equipmentId);

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _sectionTitle("Reviews", icon: Icons.reviews_rounded),
        const SizedBox(height: 10),
        StreamBuilder<QuerySnapshot<Map<String, dynamic>>>(
          stream: stream,
          builder: (context, snap) {
            if (snap.hasError) {
              return _card(
                child: Text(
                  "Review stream error: ${snap.error}",
                  style: const TextStyle(
                    color: AppColors.error,
                    fontWeight: FontWeight.w900,
                  ),
                ),
              );
            }
            if (snap.connectionState == ConnectionState.waiting) {
              return _card(
                child: Row(
                  children: [
                    const SizedBox(
                      width: 18,
                      height: 18,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    ),
                    const SizedBox(width: 10),
                    Text(
                      "Loading reviews…",
                      style: TextStyle(
                        fontWeight: FontWeight.w800,
                        color: AppColors.textDark.withOpacity(0.65),
                      ),
                    ),
                  ],
                ),
              );
            }

            final docs = snap.data?.docs ?? [];
            if (docs.isEmpty) {
              return _card(
                child: Text(
                  "No reviews yet",
                  style: TextStyle(
                    color: AppColors.textDark.withOpacity(0.6),
                    fontWeight: FontWeight.w800,
                  ),
                ),
              );
            }

            // summary
            double sum = 0;
            int good = 0;
            int bad = 0;

            for (final d in docs) {
              final m = d.data();
              final ratingNum = (m["rating"] ?? 0);
              double rating = 0;
              if (ratingNum is int) rating = ratingNum.toDouble();
              if (ratingNum is double) rating = ratingNum;
              sum += rating;

              final s = (m["sentiment"] ?? "").toString();
              if (s == "good") good++;
              if (s == "bad") bad++;
            }

            final avg = sum / docs.length;

            return Column(
              children: [
                _card(
                  child: Row(
                    children: [
                      Container(
                        padding: const EdgeInsets.all(10),
                        decoration: BoxDecoration(
                          color: AppColors.primary.withOpacity(0.14),
                          borderRadius: BorderRadius.circular(16),
                          border: Border.all(color: AppColors.border),
                        ),
                        child: const Icon(Icons.star_rounded,
                            color: AppColors.darkGreen),
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              "Average Rating",
                              style: TextStyle(
                                fontWeight: FontWeight.w800,
                                color: AppColors.textDark.withOpacity(0.70),
                              ),
                            ),
                            const SizedBox(height: 2),
                            Text(
                              "${avg.toStringAsFixed(1)} / 5.0  •  ${docs.length} review(s)",
                              style: const TextStyle(
                                fontWeight: FontWeight.w900,
                                color: AppColors.textDark,
                              ),
                            ),
                          ],
                        ),
                      ),
                      _chip(
                        icon: Icons.thumb_up_rounded,
                        text: "$good",
                        bg: const Color(0xFF15B77E).withOpacity(0.10),
                        fg: const Color(0xFF15B77E),
                        border: const Color(0xFF15B77E).withOpacity(0.22),
                      ),
                      const SizedBox(width: 8),
                      _chip(
                        icon: Icons.thumb_down_rounded,
                        text: "$bad",
                        bg: const Color(0xFFE25555).withOpacity(0.10),
                        fg: const Color(0xFFE25555),
                        border: const Color(0xFFE25555).withOpacity(0.22),
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 12),
                Column(
                  children: docs.map((d) {
                    final m = d.data();
                    final byName = (m["reviewByName"] ?? "User").toString();
                    final sentiment = (m["sentiment"] ?? "").toString();
                    final comment = (m["comment"] ?? "").toString();
                    final ratingNum = (m["rating"] ?? 0);

                    double rating = 0;
                    if (ratingNum is int) rating = ratingNum.toDouble();
                    if (ratingNum is double) rating = ratingNum;

                    final isGood = sentiment == "good";
                    final badgeColor = isGood
                        ? const Color(0xFF15B77E)
                        : const Color(0xFFE25555);

                    return Container(
                      margin: const EdgeInsets.only(bottom: 12),
                      child: _card(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Row(
                              children: [
                                Container(
                                  width: 38,
                                  height: 38,
                                  decoration: BoxDecoration(
                                    color: AppColors.surface,
                                    borderRadius: BorderRadius.circular(14),
                                    border: Border.all(color: AppColors.border),
                                  ),
                                  child: Center(
                                    child: Text(
                                      byName.trim().isNotEmpty
                                          ? byName.trim()[0].toUpperCase()
                                          : "U",
                                      style: const TextStyle(
                                        fontWeight: FontWeight.w900,
                                        color: AppColors.textDark,
                                      ),
                                    ),
                                  ),
                                ),
                                const SizedBox(width: 10),
                                Expanded(
                                  child: Text(
                                    byName,
                                    style: const TextStyle(
                                      fontWeight: FontWeight.w900,
                                      color: AppColors.textDark,
                                    ),
                                    overflow: TextOverflow.ellipsis,
                                  ),
                                ),
                                Container(
                                  padding: const EdgeInsets.symmetric(
                                      horizontal: 10, vertical: 6),
                                  decoration: BoxDecoration(
                                    color: badgeColor.withOpacity(0.10),
                                    borderRadius: BorderRadius.circular(999),
                                    border: Border.all(
                                      color: badgeColor.withOpacity(0.22),
                                    ),
                                  ),
                                  child: Text(
                                    sentiment.isEmpty
                                        ? "—"
                                        : sentiment.toUpperCase(),
                                    style: TextStyle(
                                      color: badgeColor,
                                      fontWeight: FontWeight.w900,
                                      fontSize: 11.5,
                                    ),
                                  ),
                                ),
                              ],
                            ),
                            const SizedBox(height: 10),
                            Row(
                              children: [
                                Icon(Icons.star_rounded,
                                    size: 18,
                                    color:
                                        AppColors.textDark.withOpacity(0.75)),
                                const SizedBox(width: 6),
                                Text(
                                  rating.toStringAsFixed(1),
                                  style: TextStyle(
                                    fontWeight: FontWeight.w900,
                                    color: AppColors.textDark.withOpacity(0.85),
                                  ),
                                ),
                              ],
                            ),
                            const SizedBox(height: 8),
                            Text(
                              comment.isEmpty ? "—" : comment,
                              style: TextStyle(
                                fontWeight: FontWeight.w700,
                                color: AppColors.textDark.withOpacity(0.86),
                              ),
                            ),
                          ],
                        ),
                      ),
                    );
                  }).toList(),
                ),
              ],
            );
          },
        ),
      ],
    );
  }

  Widget _topHeader() {
    final d = _equipment ?? {};

    String pick(List<String> keys) {
      for (final k in keys) {
        final v = d[k];
        if (v != null && v.toString().trim().isNotEmpty) return v.toString();
      }
      return "";
    }

    final id = pick(["Equipment_ID", "equipmentId", "id"]);
    final type = pick(["Equipment_Type", "equipment_type", "Type", "Name"]);
    final location = pick(["Main_District", "Location", "District"]);

    final displayId = id.isEmpty ? _equipmentId : id;

    return Container(
      padding: const EdgeInsets.fromLTRB(16, 14, 16, 16),
      decoration: BoxDecoration(
        gradient: AppColors.heroGradient,
        borderRadius: BorderRadius.circular(26),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.10),
            blurRadius: 24,
            offset: const Offset(0, 16),
          ),
        ],
      ),
      child: Row(
        children: [
          Container(
            width: 54,
            height: 54,
            decoration: BoxDecoration(
              color: Colors.white.withOpacity(0.18),
              borderRadius: BorderRadius.circular(18),
              border: Border.all(color: Colors.white.withOpacity(0.22)),
            ),
            child: const Icon(Icons.agriculture_rounded,
                color: Colors.white, size: 28),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  "Equipment Profile",
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 18,
                    fontWeight: FontWeight.w900,
                    letterSpacing: -0.2,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  type.isEmpty ? "—" : type,
                  style: TextStyle(
                    color: Colors.white.withOpacity(0.90),
                    fontWeight: FontWeight.w800,
                  ),
                  overflow: TextOverflow.ellipsis,
                ),
                const SizedBox(height: 10),
                Wrap(
                  spacing: 8,
                  runSpacing: 8,
                  children: [
                    _chip(
                      icon: Icons.qr_code_rounded,
                      text: displayId,
                      bg: Colors.white.withOpacity(0.14),
                      fg: Colors.white,
                      border: Colors.white.withOpacity(0.18),
                    ),
                    _chip(
                      icon: Icons.location_on_rounded,
                      text: location.isEmpty ? "—" : location,
                      bg: Colors.white.withOpacity(0.14),
                      fg: Colors.white,
                      border: Colors.white.withOpacity(0.18),
                    ),
                  ],
                ),
              ],
            ),
          ),
          const SizedBox(width: 8),
          PopupMenuButton<String>(
            icon: const Icon(Icons.more_vert_rounded, color: Colors.white),
            onSelected: (v) {
              if (v == "refresh") _load();
              if (v == "logout") _logout();
            },
            itemBuilder: (_) => const [
              PopupMenuItem(value: "refresh", child: Text("Refresh")),
              PopupMenuItem(value: "logout", child: Text("Logout")),
            ],
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final bookingStream = (_equipmentId.isEmpty)
        ? const Stream<QuerySnapshot<Map<String, dynamic>>>.empty()
        : FirebaseFirestore.instance
            .collection(EQUIPMENT_BOOKINGS_COLLECTION)
            .where("equipmentId", isEqualTo: _equipmentId)
            .snapshots();

    return Scaffold(
      backgroundColor: AppColors.surface,
      body: SafeArea(
        child: _loading
            ? const Center(child: CircularProgressIndicator())
            : _error != null
                ? Center(
                    child: Padding(
                      padding: const EdgeInsets.all(16),
                      child: _card(
                        child: Column(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            const Icon(Icons.error_rounded,
                                color: AppColors.error, size: 34),
                            const SizedBox(height: 10),
                            Text(
                              _error!,
                              textAlign: TextAlign.center,
                              style: const TextStyle(
                                color: AppColors.error,
                                fontWeight: FontWeight.w900,
                              ),
                            ),
                            const SizedBox(height: 12),
                            SizedBox(
                              height: 46,
                              width: double.infinity,
                              child: ElevatedButton.icon(
                                onPressed: _load,
                                icon: const Icon(Icons.refresh_rounded),
                                label: const Text(
                                  "Try Again",
                                  style: TextStyle(fontWeight: FontWeight.w900),
                                ),
                                style: ElevatedButton.styleFrom(
                                  backgroundColor: AppColors.primary,
                                  foregroundColor: AppColors.darkGreen,
                                  shape: RoundedRectangleBorder(
                                    borderRadius: BorderRadius.circular(999),
                                  ),
                                ),
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                  )
                : RefreshIndicator(
                    onRefresh: _load,
                    child: ListView(
                      padding: const EdgeInsets.fromLTRB(18, 18, 18, 26),
                      children: [
                        _topHeader(),
                        const SizedBox(height: 14),

                        // ✅ Equipment details (upgraded)
                        _equipmentCard(),
                        const SizedBox(height: 10),

                        // ✅ Actions row (friendly + clear)
                        Row(
                          children: [
                            Expanded(
                              child: SizedBox(
                                height: 46,
                                child: OutlinedButton.icon(
                                  onPressed: _linkMyEquipment,
                                  icon: const Icon(Icons.link_rounded),
                                  label: const Text(
                                    "Link",
                                    style:
                                        TextStyle(fontWeight: FontWeight.w900),
                                  ),
                                ),
                              ),
                            ),
                            const SizedBox(width: 10),
                            Expanded(
                              child: SizedBox(
                                height: 46,
                                child: ElevatedButton.icon(
                                  onPressed: _logout,
                                  icon: const Icon(Icons.logout_rounded),
                                  label: const Text(
                                    "Logout",
                                    style:
                                        TextStyle(fontWeight: FontWeight.w900),
                                  ),
                                  style: ElevatedButton.styleFrom(
                                    backgroundColor: AppColors.primary,
                                    foregroundColor: AppColors.darkGreen,
                                    shape: RoundedRectangleBorder(
                                      borderRadius: BorderRadius.circular(999),
                                    ),
                                  ),
                                ),
                              ),
                            ),
                          ],
                        ),

                        const SizedBox(height: 16),

                        // ✅ Reviews
                        _reviewsSection(),

                        const SizedBox(height: 18),
                        _sectionTitle("Incoming Booking Requests",
                            icon: Icons.notifications_active_rounded),
                        const SizedBox(height: 10),

                        StreamBuilder<QuerySnapshot<Map<String, dynamic>>>(
                          stream: bookingStream,
                          builder: (context, snap) {
                            if (snap.hasError) {
                              return _card(
                                child: Text(
                                  "Stream error: ${snap.error}",
                                  style: const TextStyle(
                                    color: AppColors.error,
                                    fontWeight: FontWeight.w900,
                                  ),
                                ),
                              );
                            }
                            if (snap.connectionState ==
                                ConnectionState.waiting) {
                              return _card(
                                child: Row(
                                  children: [
                                    const SizedBox(
                                      width: 18,
                                      height: 18,
                                      child: CircularProgressIndicator(
                                          strokeWidth: 2),
                                    ),
                                    const SizedBox(width: 10),
                                    Text(
                                      "Loading requests…",
                                      style: TextStyle(
                                        fontWeight: FontWeight.w800,
                                        color: AppColors.textDark
                                            .withOpacity(0.65),
                                      ),
                                    ),
                                  ],
                                ),
                              );
                            }

                            final docs = (snap.data?.docs ?? []).toList();
                            docs.sort((a, b) {
                              final at = a.data()["createdAt"];
                              final bt = b.data()["createdAt"];
                              final ad = (at is Timestamp)
                                  ? at.toDate()
                                  : DateTime(1970);
                              final bd = (bt is Timestamp)
                                  ? bt.toDate()
                                  : DateTime(1970);
                              return bd.compareTo(ad);
                            });

                            if (docs.isEmpty) {
                              return _card(
                                child: Text(
                                  "No requests yet",
                                  style: TextStyle(
                                    color: AppColors.textDark.withOpacity(0.6),
                                    fontWeight: FontWeight.w800,
                                  ),
                                ),
                              );
                            }

                            return Column(
                              children: docs.map((d) {
                                final m = d.data();
                                final status =
                                    (m["status"] ?? "pending").toString();
                                final start = (m["startDate"] ?? "").toString();
                                final end = (m["endDate"] ?? "").toString();
                                final type = (m["type"] ?? "").toString();

                                final farmerName =
                                    (m["farmerName"] ?? "Farmer").toString();
                                final farmerPhone =
                                    (m["farmerPhone"] ?? "—").toString();
                                final farmerEmail =
                                    (m["farmerEmail"] ?? "—").toString();

                                final sc = _statusColor(status);

                                return Container(
                                  margin: const EdgeInsets.only(bottom: 12),
                                  child: _card(
                                    child: Column(
                                      crossAxisAlignment:
                                          CrossAxisAlignment.start,
                                      children: [
                                        Row(
                                          children: [
                                            Container(
                                              width: 42,
                                              height: 42,
                                              decoration: BoxDecoration(
                                                color: AppColors.surface,
                                                borderRadius:
                                                    BorderRadius.circular(16),
                                                border: Border.all(
                                                    color: AppColors.border),
                                              ),
                                              child: Center(
                                                child: Text(
                                                  farmerName.trim().isNotEmpty
                                                      ? farmerName
                                                          .trim()[0]
                                                          .toUpperCase()
                                                      : "F",
                                                  style: const TextStyle(
                                                    fontWeight: FontWeight.w900,
                                                    color: AppColors.textDark,
                                                  ),
                                                ),
                                              ),
                                            ),
                                            const SizedBox(width: 10),
                                            Expanded(
                                              child: Text(
                                                farmerName,
                                                style: const TextStyle(
                                                  fontWeight: FontWeight.w900,
                                                  color: AppColors.textDark,
                                                ),
                                                overflow: TextOverflow.ellipsis,
                                              ),
                                            ),
                                            Container(
                                              padding:
                                                  const EdgeInsets.symmetric(
                                                      horizontal: 10,
                                                      vertical: 6),
                                              decoration: BoxDecoration(
                                                color: sc.withOpacity(0.10),
                                                borderRadius:
                                                    BorderRadius.circular(999),
                                                border: Border.all(
                                                  color: sc.withOpacity(0.22),
                                                ),
                                              ),
                                              child: Row(
                                                mainAxisSize: MainAxisSize.min,
                                                children: [
                                                  Icon(_statusIcon(status),
                                                      size: 14, color: sc),
                                                  const SizedBox(width: 6),
                                                  Text(
                                                    _statusText(status),
                                                    style: TextStyle(
                                                      color: sc,
                                                      fontWeight:
                                                          FontWeight.w900,
                                                      fontSize: 11.5,
                                                    ),
                                                  ),
                                                ],
                                              ),
                                            ),
                                          ],
                                        ),
                                        const SizedBox(height: 12),
                                        _infoRow(
                                          icon: Icons.phone_rounded,
                                          label: "Contact",
                                          value: farmerPhone,
                                        ),
                                        const SizedBox(height: 6),
                                        _infoRow(
                                          icon: Icons.email_rounded,
                                          label: "Email",
                                          value: farmerEmail,
                                        ),
                                        const SizedBox(height: 10),
                                        _infoRow(
                                          icon: Icons.date_range_rounded,
                                          label: "Dates",
                                          value: "$start → $end",
                                        ),
                                        const SizedBox(height: 6),
                                        _infoRow(
                                          icon: Icons.category_rounded,
                                          label: "Type",
                                          value: type.isEmpty ? "—" : type,
                                        ),
                                        if (status == "pending") ...[
                                          const SizedBox(height: 14),
                                          Row(
                                            children: [
                                              Expanded(
                                                child: SizedBox(
                                                  height: 46,
                                                  child: OutlinedButton.icon(
                                                    onPressed: () =>
                                                        _updateBookingStatus(
                                                      bookingId: d.id,
                                                      status: "rejected",
                                                    ),
                                                    icon: const Icon(
                                                      Icons.close_rounded,
                                                      color: Color(0xFFE25555),
                                                    ),
                                                    label: const Text(
                                                      "Reject",
                                                      style: TextStyle(
                                                        fontWeight:
                                                            FontWeight.w900,
                                                        color:
                                                            Color(0xFFE25555),
                                                      ),
                                                    ),
                                                    style: OutlinedButton
                                                        .styleFrom(
                                                      side: BorderSide(
                                                        color: const Color(
                                                                0xFFE25555)
                                                            .withOpacity(0.35),
                                                      ),
                                                      shape:
                                                          RoundedRectangleBorder(
                                                        borderRadius:
                                                            BorderRadius
                                                                .circular(16),
                                                      ),
                                                    ),
                                                  ),
                                                ),
                                              ),
                                              const SizedBox(width: 10),
                                              Expanded(
                                                child: SizedBox(
                                                  height: 46,
                                                  child: ElevatedButton.icon(
                                                    onPressed: () =>
                                                        _updateBookingStatus(
                                                      bookingId: d.id,
                                                      status: "accepted",
                                                    ),
                                                    icon: const Icon(
                                                        Icons.check_rounded),
                                                    label: const Text(
                                                      "Accept",
                                                      style: TextStyle(
                                                        fontWeight:
                                                            FontWeight.w900,
                                                      ),
                                                    ),
                                                    style: ElevatedButton
                                                        .styleFrom(
                                                      backgroundColor:
                                                          const Color(
                                                              0xFF15B77E),
                                                      foregroundColor:
                                                          Colors.white,
                                                      shape:
                                                          RoundedRectangleBorder(
                                                        borderRadius:
                                                            BorderRadius
                                                                .circular(16),
                                                      ),
                                                    ),
                                                  ),
                                                ),
                                              ),
                                            ],
                                          ),
                                        ],
                                      ],
                                    ),
                                  ),
                                );
                              }).toList(),
                            );
                          },
                        ),
                      ],
                    ),
                  ),
      ),
    );
  }

  static Widget _infoRow({
    required IconData icon,
    required String label,
    required String value,
  }) {
    return Row(
      children: [
        Icon(icon, size: 16, color: AppColors.textDark.withOpacity(0.65)),
        const SizedBox(width: 8),
        Text(
          "$label:",
          style: TextStyle(
            fontWeight: FontWeight.w800,
            color: AppColors.textDark.withOpacity(0.65),
          ),
        ),
        const SizedBox(width: 8),
        Expanded(
          child: Text(
            value.isEmpty ? "—" : value,
            style: const TextStyle(
              fontWeight: FontWeight.w900,
              color: AppColors.textDark,
            ),
          ),
        ),
      ],
    );
  }

  Widget _equipmentCard() {
    final d = _equipment ?? {};

    String pick(List<String> keys) {
      for (final k in keys) {
        final v = d[k];
        if (v != null && v.toString().trim().isNotEmpty) return v.toString();
      }
      return "";
    }

    final id = pick(["Equipment_ID", "equipmentId", "id"]);
    final type = pick(["Equipment_Type", "equipment_type", "Type", "Name"]);
    final crop = pick(["For_Crop", "forCrop"]);
    final hourly = pick(["Hourly_Rate_LKR", "hourlyRate"]);
    final daily = pick(["Daily_Rate_LKR", "dailyRate"]);
    final condition = pick(["Condition", "condition"]);
    final district = pick(["Main_District", "Location", "District"]);
    final nearest = pick(["Nearest_Major_District", "nearestMajorDistrict"]);
    final ownerName = pick(["Equipment_Owner_Name", "owner_name"]);
    final ownerContact = pick(["Owner_Contact_No", "owner_contact"]);
    final day = pick(["Available_Day", "availableDay"]);
    final time = pick(["Available_Time", "availableTime"]);

    final displayId = id.isEmpty ? _equipmentId : id;

    return _card(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _sectionTitle("Details", icon: Icons.info_outline_rounded),
          const SizedBox(height: 10),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: [
              _chip(icon: Icons.qr_code_rounded, text: displayId),
              if (type.isNotEmpty)
                _chip(icon: Icons.agriculture_rounded, text: type),
              if (condition.isNotEmpty)
                _chip(icon: Icons.build_circle_rounded, text: condition),
            ],
          ),
          const SizedBox(height: 12),
          _detailLine("For Crop", crop),
          _detailLine("Hourly Rate (LKR)", hourly),
          _detailLine("Daily Rate (LKR)", daily),
          _detailLine("Main District", district),
          _detailLine("Nearest District", nearest),
          _detailLine("Available Day", day),
          _detailLine("Available Time", time),
          const SizedBox(height: 12),
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: AppColors.surface,
              borderRadius: BorderRadius.circular(18),
              border: Border.all(color: AppColors.border),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  "Owner",
                  style: TextStyle(
                    fontWeight: FontWeight.w900,
                    color: AppColors.textDark,
                  ),
                ),
                const SizedBox(height: 10),
                _infoRow(
                  icon: Icons.person_rounded,
                  label: "Name",
                  value: ownerName.isEmpty ? "—" : ownerName,
                ),
                const SizedBox(height: 6),
                _infoRow(
                  icon: Icons.phone_rounded,
                  label: "Contact",
                  value: ownerContact.isEmpty ? "—" : ownerContact,
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _detailLine(String k, String v) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Row(
        children: [
          Expanded(
            child: Text(
              k,
              style: TextStyle(
                fontWeight: FontWeight.w800,
                color: AppColors.textDark.withOpacity(0.68),
              ),
            ),
          ),
          Expanded(
            child: Text(
              v.isEmpty ? "—" : v,
              textAlign: TextAlign.right,
              style: const TextStyle(
                fontWeight: FontWeight.w900,
                color: AppColors.textDark,
              ),
            ),
          ),
        ],
      ),
    );
  }
}
