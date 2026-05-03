import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

import '../../utils/app_colors.dart';
import '../../services/booking_service.dart';
import '../widgets/app_shell.dart';
import 'booking_labour_details_screen.dart';

class MyLabourBookingsScreen extends StatefulWidget {
  const MyLabourBookingsScreen({super.key});

  @override
  State<MyLabourBookingsScreen> createState() => _MyLabourBookingsScreenState();
}

class _MyLabourBookingsScreenState extends State<MyLabourBookingsScreen> {
  Future<void> _onRefresh() async {
    setState(() {});
    await Future<void>.delayed(const Duration(milliseconds: 350));
  }

  static Color _statusColor(String s) {
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

  @override
  Widget build(BuildContext context) {
    return AppShell(
      currentIndex: 0,
      child: Scaffold(
        backgroundColor: const Color(0xFFF9FDF2),
        body: SafeArea(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              _heroHeader(context),
              Expanded(
                child: StreamBuilder<QuerySnapshot<Map<String, dynamic>>>(
                  stream: BookingService.myBookings(),
                  builder: (context, snap) {
                    if (snap.hasError) {
                      return RefreshIndicator(
                        onRefresh: _onRefresh,
                        child: CustomScrollView(
                          physics: const AlwaysScrollableScrollPhysics(),
                          slivers: [
                            SliverFillRemaining(
                              hasScrollBody: false,
                              child: _errorState(snap.error.toString()),
                            ),
                          ],
                        ),
                      );
                    }
                    if (snap.connectionState == ConnectionState.waiting) {
                      return RefreshIndicator(
                        onRefresh: _onRefresh,
                        child: CustomScrollView(
                          physics: const AlwaysScrollableScrollPhysics(),
                          slivers: [
                            SliverPadding(
                              padding: const EdgeInsets.fromLTRB(16, 12, 16, 100),
                              sliver: SliverList(
                                delegate: SliverChildBuilderDelegate(
                                  (_, __) => const Padding(
                                    padding: EdgeInsets.only(bottom: 12),
                                    child: _BookingSkeletonCard(),
                                  ),
                                  childCount: 4,
                                ),
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
                      final ad = (at is Timestamp) ? at.toDate() : DateTime(0);
                      final bd = (bt is Timestamp) ? bt.toDate() : DateTime(0);
                      return bd.compareTo(ad);
                    });
                    if (docs.isEmpty) {
                      return RefreshIndicator(
                        onRefresh: _onRefresh,
                        child: CustomScrollView(
                          physics: const AlwaysScrollableScrollPhysics(),
                          slivers: [
                            SliverFillRemaining(
                              hasScrollBody: false,
                              child: _emptyState(context),
                            ),
                          ],
                        ),
                      );
                    }
                    return RefreshIndicator(
                      onRefresh: _onRefresh,
                      child: ListView.builder(
                        physics: const AlwaysScrollableScrollPhysics(
                          parent: BouncingScrollPhysics(),
                        ),
                        padding: const EdgeInsets.fromLTRB(16, 12, 16, 100),
                        itemCount: docs.length,
                        itemBuilder: (context, i) {
                          final d = docs[i];
                          final m = d.data();
                          final status = (m["status"] ?? "pending").toString();
                          final labourId = (m["labourId"] ?? "").toString();
                          final start = (m["startDate"] ?? "").toString();
                          final end = (m["endDate"] ?? "").toString();
                          final type = (m["type"] ?? "").toString();
                          final labourUid = (m["labourUid"] ?? "").toString();
                          return Padding(
                            padding: const EdgeInsets.only(bottom: 12),
                            child: _BookingCard(
                              title: "Labour: $labourId",
                              status: status,
                              start: start,
                              end: end,
                              type: type,
                              statusColor: _statusColor(status),
                              leadingIcon: Icons.groups_rounded,
                              onCancel: status == "pending"
                                  ? () async {
                                      await BookingService.updateStatus(
                                        bookingId: d.id,
                                        status: "cancelled",
                                      );
                                    }
                                  : null,
                              acceptedHint: "Tap to view details",
                              onTap: status == "accepted"
                                  ? () {
                                      Navigator.push(
                                        context,
                                        MaterialPageRoute(
                                          builder: (_) =>
                                              BookingLabourDetailsScreen(
                                            labourId: labourId,
                                            labourUid: labourUid,
                                            startDate: start,
                                            endDate: end,
                                            type: type,
                                          ),
                                        ),
                                      );
                                    }
                                  : null,
                            ),
                          );
                        },
                      ),
                    );
                  },
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _heroHeader(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.fromLTRB(16, 12, 16, 14),
      decoration: const BoxDecoration(
        gradient: AppColors.heroGradient,
        borderRadius: BorderRadius.only(
          bottomLeft: Radius.circular(28),
          bottomRight: Radius.circular(28),
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Row(
            children: [
              _roundIconBtn(
                icon: Icons.arrow_back_ios_new_rounded,
                onTap: () => Navigator.pop(context),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  "Labour bookings",
                  style: GoogleFonts.poppins(
                    color: Colors.white.withOpacity(0.95),
                    fontWeight: FontWeight.w700,
                    fontSize: 16,
                    letterSpacing: 0.15,
                  ),
                ),
              ),
              _roundIconBtn(
                icon: Icons.refresh_rounded,
                onTap: _onRefresh,
              ),
            ],
          ),
          const SizedBox(height: 6),
          Align(
            alignment: Alignment.centerLeft,
            child: Text(
              "Your scheduled work",
              style: GoogleFonts.poppins(
                color: Colors.white.withOpacity(0.97),
                fontWeight: FontWeight.w600,
                fontSize: 21,
                height: 1.12,
                letterSpacing: -0.2,
              ),
            ),
          ),
          const SizedBox(height: 2),
          Text(
            "Bookings you made · Hire Labour",
            style: GoogleFonts.inter(
              color: Colors.white.withOpacity(0.82),
              fontWeight: FontWeight.w500,
              fontSize: 12.5,
              height: 1.35,
            ),
          ),
        ],
      ),
    );
  }

  Widget _roundIconBtn({
    required IconData icon,
    required VoidCallback onTap,
  }) {
    return InkWell(
      borderRadius: BorderRadius.circular(999),
      onTap: onTap,
      child: CircleAvatar(
        radius: 18,
        backgroundColor: Colors.white.withOpacity(0.14),
        child: Icon(icon, color: Colors.white, size: 20),
      ),
    );
  }

  Widget _emptyState(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(24, 24, 24, 24),
      child: Center(
        child: Container(
          padding: const EdgeInsets.all(20),
          decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.circular(18),
            border: Border.all(color: AppColors.border),
            boxShadow: [
              BoxShadow(
                color: Colors.black.withOpacity(0.05),
                blurRadius: 12,
                offset: const Offset(0, 6),
              ),
            ],
          ),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              CircleAvatar(
                radius: 36,
                backgroundColor: AppColors.primary.withOpacity(0.2),
                child: Icon(
                  Icons.calendar_today_rounded,
                  size: 36,
                  color: AppColors.darkGreen.withOpacity(0.85),
                ),
              ),
              const SizedBox(height: 16),
              Text(
                "No labour bookings yet",
                textAlign: TextAlign.center,
                style: GoogleFonts.poppins(
                  fontWeight: FontWeight.w700,
                  fontSize: 17,
                  color: AppColors.textDark,
                ),
              ),
              const SizedBox(height: 8),
              Text(
                "Find workers from Hire Labour on the home dashboard.",
                textAlign: TextAlign.center,
                style: GoogleFonts.inter(
                  fontWeight: FontWeight.w500,
                  fontSize: 13.2,
                  height: 1.4,
                  color: AppColors.textDark.withOpacity(0.65),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _errorState(String message) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(24, 24, 24, 24),
      child: Center(
        child: Container(
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: Colors.red.withOpacity(0.06),
            borderRadius: BorderRadius.circular(18),
            border: Border.all(color: Colors.red.withOpacity(0.2)),
          ),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                "Couldn’t load bookings",
                style: GoogleFonts.poppins(
                  fontWeight: FontWeight.w700,
                  fontSize: 17,
                  color: Colors.red,
                ),
              ),
              const SizedBox(height: 8),
              Text(
                message,
                style: GoogleFonts.inter(
                  fontWeight: FontWeight.w500,
                  fontSize: 13.2,
                  height: 1.35,
                  color: Colors.red.withOpacity(0.88),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _BookingSkeletonCard extends StatelessWidget {
  const _BookingSkeletonCard();

  Widget _bar({double w = double.infinity, double h = 12}) {
    return Container(
      width: w,
      height: h,
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: AppColors.border),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: AppColors.border),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.06),
            blurRadius: 20,
            offset: const Offset(0, 12),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              _bar(w: 140, h: 14),
              const Spacer(),
              _bar(w: 72, h: 26),
            ],
          ),
          const SizedBox(height: 12),
          _bar(),
          const SizedBox(height: 8),
          _bar(w: 200),
        ],
      ),
    );
  }
}

class _BookingCard extends StatelessWidget {
  final String title;
  final String status;
  final String start;
  final String end;
  final String type;
  final Color statusColor;
  final IconData leadingIcon;
  final VoidCallback? onCancel;
  final String acceptedHint;
  final VoidCallback? onTap;

  const _BookingCard({
    required this.title,
    required this.status,
    required this.start,
    required this.end,
    required this.type,
    required this.statusColor,
    required this.leadingIcon,
    this.onCancel,
    required this.acceptedHint,
    this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(18),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.06),
            blurRadius: 20,
            offset: const Offset(0, 12),
          ),
        ],
      ),
      child: Material(
        color: Colors.white,
        borderRadius: BorderRadius.circular(18),
        child: InkWell(
          borderRadius: BorderRadius.circular(18),
          onTap: onTap,
          child: Padding(
            padding: const EdgeInsets.all(14),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    CircleAvatar(
                      radius: 22,
                      backgroundColor: AppColors.primary.withOpacity(0.16),
                      child: Icon(
                        leadingIcon,
                        color: AppColors.darkGreen,
                        size: 22,
                      ),
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            title,
                            maxLines: 2,
                            overflow: TextOverflow.ellipsis,
                            style: GoogleFonts.poppins(
                              fontWeight: FontWeight.w700,
                              fontSize: 15,
                              letterSpacing: -0.15,
                              color: AppColors.textDark,
                            ),
                          ),
                        ],
                      ),
                    ),
                    const SizedBox(width: 8),
                    Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 10, vertical: 6),
                      decoration: BoxDecoration(
                        color: statusColor.withOpacity(0.12),
                        borderRadius: BorderRadius.circular(999),
                        border: Border.all(
                            color: statusColor.withOpacity(0.28)),
                      ),
                      child: Text(
                        status.toUpperCase(),
                        style: GoogleFonts.inter(
                          color: statusColor,
                          fontWeight: FontWeight.w700,
                          fontSize: 10.5,
                          letterSpacing: 0.4,
                        ),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 12),
                Row(
                  children: [
                    Icon(
                      Icons.date_range_rounded,
                      size: 17,
                      color: AppColors.darkGreen,
                    ),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text(
                        "$start → $end",
                        style: GoogleFonts.inter(
                          fontWeight: FontWeight.w500,
                          fontSize: 12.8,
                          color: AppColors.textDark.withOpacity(0.75),
                        ),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 8),
                Row(
                  children: [
                    Icon(
                      Icons.category_rounded,
                      size: 17,
                      color: AppColors.darkGreen,
                    ),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text(
                        type.isEmpty ? "—" : type,
                        style: GoogleFonts.inter(
                          fontWeight: FontWeight.w500,
                          fontSize: 12.8,
                          color: AppColors.textDark.withOpacity(0.75),
                        ),
                      ),
                    ),
                    if (onCancel != null)
                      TextButton(
                        onPressed: onCancel,
                        style: TextButton.styleFrom(
                          padding: const EdgeInsets.symmetric(
                              horizontal: 8, vertical: 4),
                          minimumSize: Size.zero,
                          tapTargetSize: MaterialTapTargetSize.shrinkWrap,
                        ),
                        child: Text(
                          "Cancel",
                          style: GoogleFonts.poppins(
                            fontWeight: FontWeight.w700,
                            fontSize: 12,
                            color: AppColors.error,
                          ),
                        ),
                      ),
                  ],
                ),
                if (status == "accepted") ...[
                  const SizedBox(height: 12),
                  Container(
                    width: double.infinity,
                    padding: const EdgeInsets.symmetric(
                        horizontal: 12, vertical: 10),
                    decoration: BoxDecoration(
                      color: AppColors.primary.withOpacity(0.10),
                      borderRadius: BorderRadius.circular(14),
                      border: Border.all(
                        color: AppColors.primary.withOpacity(0.35),
                      ),
                    ),
                    child: Row(
                      children: [
                        Icon(
                          Icons.touch_app_rounded,
                          size: 18,
                          color: AppColors.darkGreen.withOpacity(0.95),
                        ),
                        const SizedBox(width: 8),
                        Expanded(
                          child: Text(
                            acceptedHint,
                            style: GoogleFonts.inter(
                              fontWeight: FontWeight.w600,
                              color: AppColors.textDark.withOpacity(0.82),
                              fontSize: 12.5,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ],
            ),
          ),
        ),
      ),
    );
  }
}
