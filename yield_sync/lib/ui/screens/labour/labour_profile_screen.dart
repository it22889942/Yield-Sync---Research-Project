import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';

import '../../../utils/app_colors.dart';
import '../../../services/auth_service.dart';
import '../../../services/booking_service.dart';
import '../../../services/review_service.dart';

class LabourProfileScreen extends StatefulWidget {
  const LabourProfileScreen({super.key});

  @override
  State<LabourProfileScreen> createState() => _LabourProfileScreenState();
}

class _LabourProfileScreenState extends State<LabourProfileScreen> {
  bool _loading = true;
  String? _error;

  Map<String, dynamic>? _labour;
  String _labourId = "";

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
      final labourId = await AuthService().getMyLabourId();
      if (labourId == null || labourId.trim().isEmpty) {
        throw Exception("Labour ID not linked to this user.");
      }

      final snap = await FirebaseFirestore.instance
          .collection("labours")
          .doc(labourId.trim())
          .get();

      if (!snap.exists) {
        throw Exception("Labour details not found for ID: $labourId");
      }

      if (!mounted) return;
      setState(() {
        _labourId = labourId.trim();
        _labour = snap.data();
      });
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  // ✅ Modern status palette
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

  String _statusText(String s) =>
      (s.trim().isEmpty ? "pending" : s).toUpperCase();

  Future<void> _logout() async {
    try {
      await FirebaseAuth.instance.signOut();
      if (!mounted) return;
      Navigator.of(context).pushNamedAndRemoveUntil("/login", (r) => false);
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("Logout failed: $e")),
      );
    }
  }

  // ✅ Accepted booking popup (modern)
  Future<void> _showAcceptedBookingPopup({
    required Map<String, dynamic> booking,
    required String bookingId,
  }) async {
    final farmerName = (booking["farmerName"] ?? "").toString();
    final farmerPhone = (booking["farmerPhone"] ?? "").toString();
    final farmerEmail = (booking["farmerEmail"] ?? "").toString();

    final start = (booking["startDate"] ?? "").toString();
    final end = (booking["endDate"] ?? "").toString();
    final type = (booking["type"] ?? "").toString();
    final status = (booking["status"] ?? "").toString();
    final note = (booking["note"] ?? "").toString();

    showDialog(
      context: context,
      barrierDismissible: true,
      builder: (_) {
        return Dialog(
          insetPadding: const EdgeInsets.all(14),
          backgroundColor: const Color(0xFFF2EEF6),
          shape:
              RoundedRectangleBorder(borderRadius: BorderRadius.circular(22)),
          child: Padding(
            padding: const EdgeInsets.fromLTRB(16, 16, 16, 14),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Container(
                      padding: const EdgeInsets.all(10),
                      decoration: BoxDecoration(
                        color: AppColors.primary.withOpacity(0.16),
                        borderRadius: BorderRadius.circular(16),
                        border: Border.all(color: AppColors.primary, width: 1),
                      ),
                      child: const Icon(Icons.event_available_rounded,
                          color: AppColors.darkGreen),
                    ),
                    const SizedBox(width: 12),
                    const Expanded(
                      child: Text(
                        "Accepted Booking",
                        style: TextStyle(
                          fontWeight: FontWeight.w900,
                          fontSize: 18,
                          color: AppColors.textDark,
                        ),
                      ),
                    ),
                    IconButton(
                      onPressed: () => Navigator.pop(context),
                      icon: const Icon(Icons.close_rounded),
                    ),
                  ],
                ),
                const SizedBox(height: 10),
                _card(
                  child: Column(
                    children: [
                      _kv("Booking ID", bookingId),
                      _kv("Status", _statusText(status)),
                      _kv("Dates", "$start → $end"),
                      _kv("Type", type.isEmpty ? "—" : type),
                      if (note.trim().isNotEmpty) _kv("Note", note.trim()),
                    ],
                  ),
                ),
                const SizedBox(height: 12),
                _card(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text(
                        "Farmer Details",
                        style: TextStyle(
                          fontWeight: FontWeight.w900,
                          color: AppColors.textDark,
                        ),
                      ),
                      const SizedBox(height: 10),
                      _infoRow(
                        icon: Icons.person_rounded,
                        label: "Name",
                        value: farmerName.isNotEmpty ? farmerName : "Farmer",
                      ),
                      const SizedBox(height: 6),
                      _infoRow(
                        icon: Icons.phone_rounded,
                        label: "Contact",
                        value: farmerPhone.isNotEmpty ? farmerPhone : "—",
                      ),
                      const SizedBox(height: 6),
                      _infoRow(
                        icon: Icons.email_rounded,
                        label: "Email",
                        value: farmerEmail.isNotEmpty ? farmerEmail : "—",
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 14),
                SizedBox(
                  width: double.infinity,
                  height: 46,
                  child: ElevatedButton.icon(
                    onPressed: () => Navigator.pop(context),
                    icon: const Icon(Icons.close_rounded),
                    label: const Text(
                      "Close",
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
        );
      },
    );
  }

  // ✅ Modern card helper
  static Widget _card({required Widget child, EdgeInsets? padding}) {
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

  static Widget _kv(String k, String v) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Row(
        children: [
          Expanded(
            child: Text(
              k,
              style: TextStyle(
                fontWeight: FontWeight.w800,
                color: AppColors.textDark.withOpacity(0.70),
              ),
            ),
          ),
          Expanded(
            child: Text(
              v,
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

  // ✅ Reviews (modern + summary)
  Widget _reviewsSection() {
    final stream = (_labourId.isEmpty)
        ? const Stream<QuerySnapshot<Map<String, dynamic>>>.empty()
        : ReviewService.labourReviews(_labourId);

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

            double sum = 0;
            int good = 0;
            int bad = 0;

            for (final d in docs) {
              final m = d.data();
              final ratingNum = (m["rating"] ?? 0);
              double r = 0;
              if (ratingNum is int) r = ratingNum.toDouble();
              if (ratingNum is double) r = ratingNum;
              sum += r;

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
                      _miniStat(
                        icon: Icons.thumb_up_rounded,
                        value: "$good",
                        color: const Color(0xFF15B77E),
                      ),
                      const SizedBox(width: 8),
                      _miniStat(
                        icon: Icons.thumb_down_rounded,
                        value: "$bad",
                        color: const Color(0xFFE25555),
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

  static Widget _miniStat({
    required IconData icon,
    required String value,
    required Color color,
  }) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 7),
      decoration: BoxDecoration(
        color: color.withOpacity(0.10),
        borderRadius: BorderRadius.circular(999),
        border: Border.all(color: color.withOpacity(0.22)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 16, color: color),
          const SizedBox(width: 6),
          Text(
            value,
            style: TextStyle(
              fontWeight: FontWeight.w900,
              color: color,
            ),
          ),
        ],
      ),
    );
  }

  Widget _topHeader() {
    final d = _labour ?? {};
    final id = (d["Labour_ID"] ?? d["labourId"] ?? _labourId).toString();
    final name = (d["Name"] ?? "").toString();
    final type = (d["Labour_Type"] ?? "").toString();
    final location = (d["Location"] ?? "").toString();

    final titleName = name.trim().isEmpty ? "Labour" : name.trim();

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
            child: const Icon(Icons.handyman_rounded,
                color: Colors.white, size: 28),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  "Labour Profile",
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 18,
                    fontWeight: FontWeight.w900,
                    letterSpacing: -0.2,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  titleName,
                  style: TextStyle(
                    color: Colors.white.withOpacity(0.92),
                    fontWeight: FontWeight.w800,
                  ),
                  overflow: TextOverflow.ellipsis,
                ),
                const SizedBox(height: 10),
                Wrap(
                  spacing: 8,
                  runSpacing: 8,
                  children: [
                    _headerChip(Icons.badge_rounded, id),
                    _headerChip(Icons.work_rounded, type.isEmpty ? "—" : type),
                    _headerChip(Icons.location_on_rounded,
                        location.isEmpty ? "—" : location),
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

  static Widget _headerChip(IconData icon, String text) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 7),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.14),
        borderRadius: BorderRadius.circular(999),
        border: Border.all(color: Colors.white.withOpacity(0.18)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 16, color: Colors.white),
          const SizedBox(width: 6),
          ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 170),
            child: Text(
              text.isEmpty ? "—" : text,
              style: const TextStyle(
                color: Colors.white,
                fontWeight: FontWeight.w800,
              ),
              overflow: TextOverflow.ellipsis,
            ),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final bookingStream = (_labourId.isEmpty)
        ? const Stream<QuerySnapshot<Map<String, dynamic>>>.empty()
        : BookingService.labourBookings(labourId: _labourId);

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
                                fontWeight: FontWeight.w900,
                                color: AppColors.error,
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
                        _labourCard(),
                        const SizedBox(height: 16),
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
                                final note = (m["note"] ?? "").toString();

                                final farmerName =
                                    (m["farmerName"] ?? "").toString();
                                final farmerPhone =
                                    (m["farmerPhone"] ?? "").toString();
                                final farmerEmail =
                                    (m["farmerEmail"] ?? "").toString();

                                final sc = _statusColor(status);
                                final clickableAccepted = status == "accepted";

                                return InkWell(
                                  borderRadius: BorderRadius.circular(22),
                                  onTap: clickableAccepted
                                      ? () => _showAcceptedBookingPopup(
                                            booking: m,
                                            bookingId: d.id,
                                          )
                                      : null,
                                  child: Container(
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
                                                      fontWeight:
                                                          FontWeight.w900,
                                                      color: AppColors.textDark,
                                                    ),
                                                  ),
                                                ),
                                              ),
                                              const SizedBox(width: 10),
                                              Expanded(
                                                child: Text(
                                                  farmerName.isNotEmpty
                                                      ? farmerName
                                                      : "Farmer",
                                                  style: const TextStyle(
                                                    fontWeight: FontWeight.w900,
                                                    color: AppColors.textDark,
                                                  ),
                                                  overflow:
                                                      TextOverflow.ellipsis,
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
                                                      BorderRadius.circular(
                                                          999),
                                                  border: Border.all(
                                                    color: sc.withOpacity(0.22),
                                                  ),
                                                ),
                                                child: Row(
                                                  mainAxisSize:
                                                      MainAxisSize.min,
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
                                            value: farmerPhone.isNotEmpty
                                                ? farmerPhone
                                                : "—",
                                          ),
                                          const SizedBox(height: 6),
                                          _infoRow(
                                            icon: Icons.email_rounded,
                                            label: "Email",
                                            value: farmerEmail.isNotEmpty
                                                ? farmerEmail
                                                : "—",
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
                                          if (note.trim().isNotEmpty) ...[
                                            const SizedBox(height: 10),
                                            Container(
                                              padding: const EdgeInsets.all(12),
                                              decoration: BoxDecoration(
                                                color: AppColors.surface,
                                                borderRadius:
                                                    BorderRadius.circular(18),
                                                border: Border.all(
                                                    color: AppColors.border),
                                              ),
                                              child: Row(
                                                children: [
                                                  Icon(
                                                      Icons
                                                          .sticky_note_2_rounded,
                                                      size: 18,
                                                      color: AppColors.textDark
                                                          .withOpacity(0.60)),
                                                  const SizedBox(width: 8),
                                                  Expanded(
                                                    child: Text(
                                                      note.trim(),
                                                      style: TextStyle(
                                                        fontWeight:
                                                            FontWeight.w700,
                                                        color: AppColors
                                                            .textDark
                                                            .withOpacity(0.80),
                                                      ),
                                                    ),
                                                  ),
                                                ],
                                              ),
                                            ),
                                          ],
                                          if (clickableAccepted) ...[
                                            const SizedBox(height: 12),
                                            Container(
                                              padding: const EdgeInsets.all(12),
                                              decoration: BoxDecoration(
                                                color: AppColors.primary
                                                    .withOpacity(0.10),
                                                borderRadius:
                                                    BorderRadius.circular(18),
                                                border: Border.all(
                                                    color: AppColors.border),
                                              ),
                                              child: Row(
                                                children: [
                                                  const Icon(
                                                      Icons.touch_app_rounded,
                                                      size: 18,
                                                      color:
                                                          AppColors.darkGreen),
                                                  const SizedBox(width: 8),
                                                  Expanded(
                                                    child: Text(
                                                      "Tap to view full details",
                                                      style: TextStyle(
                                                        fontWeight:
                                                            FontWeight.w800,
                                                        color: AppColors
                                                            .textDark
                                                            .withOpacity(0.70),
                                                      ),
                                                    ),
                                                  ),
                                                ],
                                              ),
                                            ),
                                          ],
                                          if (status == "pending") ...[
                                            const SizedBox(height: 14),
                                            Row(
                                              children: [
                                                Expanded(
                                                  child: SizedBox(
                                                    height: 46,
                                                    child: OutlinedButton.icon(
                                                      onPressed: () async {
                                                        await BookingService
                                                            .updateStatus(
                                                          bookingId: d.id,
                                                          status: "rejected",
                                                        );
                                                      },
                                                      icon: const Icon(
                                                        Icons.close_rounded,
                                                        color:
                                                            Color(0xFFE25555),
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
                                                              .withOpacity(
                                                                  0.35),
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
                                                      onPressed: () async {
                                                        await BookingService
                                                            .updateStatus(
                                                          bookingId: d.id,
                                                          status: "accepted",
                                                        );
                                                      },
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

  Widget _labourCard() {
    final d = _labour ?? {};

    final id = (d["Labour_ID"] ?? d["labourId"] ?? _labourId).toString();
    final name = (d["Name"] ?? "").toString();
    final type = (d["Labour_Type"] ?? "").toString();
    final rate = (d["Hourly_Rate"] ?? "").toString();
    final location = (d["Location"] ?? "").toString();

    return _card(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _sectionTitle("Details", icon: Icons.info_outline_rounded),
          const SizedBox(height: 10),
          _kv("Labour ID", id.isEmpty ? "—" : id),
          _kv("Name", name.isEmpty ? "—" : name),
          _kv("Type", type.isEmpty ? "—" : type),
          _kv("Rate", rate.isEmpty ? "—" : rate),
          _kv("Location", location.isEmpty ? "—" : location),
        ],
      ),
    );
  }
}
