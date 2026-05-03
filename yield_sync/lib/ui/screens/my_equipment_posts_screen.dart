import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/material.dart';

import '../../utils/app_colors.dart';
import '../../services/app_routes.dart';
import '../../services/labour_equipment_post_service.dart';
import '../widgets/app_shell.dart';

class MyEquipmentPostsScreen extends StatelessWidget {
  const MyEquipmentPostsScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return AppShell(
      currentIndex: 0,
      child: Scaffold(
        backgroundColor: AppColors.surface,
        body: SafeArea(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              _header(context),
              Expanded(
                child: StreamBuilder<QuerySnapshot<Map<String, dynamic>>>(
                  stream: LabourEquipmentPostService.myEquipmentPostsStream(),
                  builder: (context, snap) {
                    if (snap.hasError) {
                      return Center(
                        child: Padding(
                          padding: const EdgeInsets.all(24),
                          child: Text(
                            "Error: ${snap.error}",
                            textAlign: TextAlign.center,
                            style: TextStyle(
                              color: AppColors.error,
                              fontWeight: FontWeight.w800,
                            ),
                          ),
                        ),
                      );
                    }
                    if (snap.connectionState == ConnectionState.waiting) {
                      return const Center(child: CircularProgressIndicator());
                    }
                    final docs = snap.data?.docs ?? [];
                    final sorted = List<QueryDocumentSnapshot<Map<String, dynamic>>>.from(docs);
                    sorted.sort((a, b) {
                      final at = a.data()["createdAt"];
                      final bt = b.data()["createdAt"];
                      if (at == null && bt == null) return 0;
                      if (at == null) return 1;
                      if (bt == null) return -1;
                      final ad = (at is Timestamp) ? at.toDate() : DateTime(0);
                      final bd = (bt is Timestamp) ? bt.toDate() : DateTime(0);
                      return bd.compareTo(ad);
                    });
                    if (sorted.isEmpty) {
                      return Center(
                        child: Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Icon(Icons.agriculture_rounded,
                                size: 64, color: AppColors.textDark.withOpacity(0.3)),
                            const SizedBox(height: 16),
                            Text(
                              "No equipment ads yet",
                              style: TextStyle(
                                fontWeight: FontWeight.w800,
                                color: AppColors.textDark.withOpacity(0.65),
                              ),
                            ),
                            const SizedBox(height: 8),
                            Text(
                              "Add one from the home menu",
                              style: TextStyle(
                                fontWeight: FontWeight.w600,
                                color: AppColors.textDark.withOpacity(0.5),
                                fontSize: 13,
                              ),
                            ),
                          ],
                        ),
                      );
                    }
                    return ListView.builder(
                      padding: const EdgeInsets.fromLTRB(16, 12, 16, 100),
                      itemCount: sorted.length,
                      itemBuilder: (context, i) {
                        final d = sorted[i];
                        final m = d.data();
                        final id = (m["Equipment_ID"] ?? m["equipmentId"] ?? d.id).toString();
                        final type = (m["Equipment_Type"] ?? m["equipmentType"] ?? "").toString();
                        final district = (m["Nearest_Major_District"] ?? m["Main_District"] ?? "").toString();
                        final daily = (m["Daily_Rate_LKR"] ?? m["dailyRate"] ?? 0);
                        final dailyRate = (daily is num) ? daily.toDouble() : double.tryParse(daily.toString()) ?? 0.0;
                        final mod =
                            LabourEquipmentPostService.moderationStateFromData(m);
                        return _PostCard(
                          equipmentId: id,
                          equipmentType: type,
                          district: district,
                          dailyRate: dailyRate,
                          moderation: mod,
                          onOpen: () => Navigator.pushNamed(
                            context,
                            AppRoutes.equipmentDetails,
                            arguments: id,
                          ),
                        );
                      },
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

  Widget _header(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.fromLTRB(16, 14, 16, 18),
      decoration: const BoxDecoration(
        gradient: AppColors.heroGradient,
        borderRadius: BorderRadius.only(
          bottomLeft: Radius.circular(28),
          bottomRight: Radius.circular(28),
        ),
      ),
      child: Row(
        children: [
          IconButton(
            onPressed: () => Navigator.pop(context),
            icon: Icon(Icons.arrow_back_rounded,
                color: Colors.white.withOpacity(0.95)),
          ),
          const SizedBox(width: 8),
          const Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  "Your equipment ads",
                  style: TextStyle(
                    color: Colors.white,
                    fontWeight: FontWeight.w900,
                    fontSize: 18,
                  ),
                ),
                Text(
                  "Posts you created",
                  style: TextStyle(
                    color: Colors.white70,
                    fontWeight: FontWeight.w600,
                    fontSize: 12,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _PostCard extends StatelessWidget {
  final String equipmentId;
  final String equipmentType;
  final String district;
  final double dailyRate;
  final PostModerationState moderation;
  final VoidCallback onOpen;

  const _PostCard({
    required this.equipmentId,
    required this.equipmentType,
    required this.district,
    required this.dailyRate,
    required this.moderation,
    required this.onOpen,
  });

  static const _iconBg = Color(0xFF1C9BE8);

  void _onCardTap(BuildContext context) {
    switch (moderation) {
      case PostModerationState.pending:
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text(
              'Under review. This ad is hidden from search until an admin approves it.',
            ),
          ),
        );
        return;
      case PostModerationState.rejected:
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('This ad was not approved and stays off public listings.'),
          ),
        );
        return;
      case PostModerationState.approved:
        onOpen();
    }
  }

  @override
  Widget build(BuildContext context) {
    final pending = moderation == PostModerationState.pending;
    final rejected = moderation == PostModerationState.rejected;
    final dimmed = pending || rejected;
    final title = equipmentType.isEmpty ? equipmentId : equipmentType;

    Widget card = Container(
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(
          color: pending
              ? const Color(0xFFF59E0B).withOpacity(0.45)
              : rejected
                  ? AppColors.error.withOpacity(0.35)
                  : AppColors.border,
        ),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.05),
            blurRadius: 12,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Material(
        color: Colors.transparent,
        borderRadius: BorderRadius.circular(20),
        child: InkWell(
          borderRadius: BorderRadius.circular(20),
          onTap: () => _onCardTap(context),
          child: Padding(
            padding: const EdgeInsets.all(18),
            child: Row(
              children: [
                Container(
                  width: 56,
                  height: 56,
                  decoration: BoxDecoration(
                    color: _iconBg.withOpacity(dimmed ? 0.45 : 1),
                    borderRadius: BorderRadius.circular(16),
                  ),
                  child: Icon(
                    pending
                        ? Icons.hourglass_top_rounded
                        : rejected
                            ? Icons.block_rounded
                            : Icons.agriculture_rounded,
                    color: Colors.white.withOpacity(dimmed ? 0.9 : 1),
                    size: 28,
                  ),
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        children: [
                          Expanded(
                            child: Text(
                              title,
                              style: TextStyle(
                                fontWeight: FontWeight.w900,
                                color: AppColors.textDark.withOpacity(dimmed ? 0.55 : 1),
                                fontSize: 16,
                              ),
                              maxLines: 1,
                              overflow: TextOverflow.ellipsis,
                            ),
                          ),
                          if (pending) ...[
                            const SizedBox(width: 8),
                            Container(
                              padding: const EdgeInsets.symmetric(
                                horizontal: 8,
                                vertical: 4,
                              ),
                              decoration: BoxDecoration(
                                color: const Color(0xFFFEF3C7),
                                borderRadius: BorderRadius.circular(8),
                                border: Border.all(
                                  color: const Color(0xFFF59E0B).withOpacity(0.5),
                                ),
                              ),
                              child: const Text(
                                'Under review',
                                style: TextStyle(
                                  fontWeight: FontWeight.w800,
                                  fontSize: 11,
                                  color: Color(0xFFB45309),
                                ),
                              ),
                            ),
                          ],
                          if (rejected) ...[
                            const SizedBox(width: 8),
                            Container(
                              padding: const EdgeInsets.symmetric(
                                horizontal: 8,
                                vertical: 4,
                              ),
                              decoration: BoxDecoration(
                                color: AppColors.error.withOpacity(0.1),
                                borderRadius: BorderRadius.circular(8),
                              ),
                              child: Text(
                                'Not approved',
                                style: TextStyle(
                                  fontWeight: FontWeight.w800,
                                  fontSize: 11,
                                  color: AppColors.error,
                                ),
                              ),
                            ),
                          ],
                        ],
                      ),
                      if (district.isNotEmpty) ...[
                        const SizedBox(height: 6),
                        Row(
                          children: [
                            Icon(
                              Icons.location_on_rounded,
                              size: 14,
                              color: AppColors.textDark.withOpacity(dimmed ? 0.35 : 0.5),
                            ),
                            const SizedBox(width: 4),
                            Expanded(
                              child: Text(
                                district,
                                style: TextStyle(
                                  fontWeight: FontWeight.w600,
                                  color: AppColors.textDark.withOpacity(dimmed ? 0.38 : 0.55),
                                  fontSize: 12,
                                ),
                                maxLines: 1,
                                overflow: TextOverflow.ellipsis,
                              ),
                            ),
                          ],
                        ),
                      ],
                    ],
                  ),
                ),
                const SizedBox(width: 12),
                Column(
                  crossAxisAlignment: CrossAxisAlignment.end,
                  children: [
                    Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 10,
                        vertical: 6,
                      ),
                      decoration: BoxDecoration(
                        color: AppColors.primary.withOpacity(dimmed ? 0.06 : 0.12),
                        borderRadius: BorderRadius.circular(10),
                      ),
                      child: Text(
                        "LKR ${dailyRate.toStringAsFixed(0)}/day",
                        style: TextStyle(
                          fontWeight: FontWeight.w900,
                          color: AppColors.darkGreen.withOpacity(dimmed ? 0.55 : 1),
                          fontSize: 13,
                        ),
                      ),
                    ),
                    const SizedBox(height: 10),
                    Container(
                      width: 36,
                      height: 36,
                      decoration: BoxDecoration(
                        color: AppColors.textDark.withOpacity(dimmed ? 0.05 : 0.08),
                        shape: BoxShape.circle,
                      ),
                      child: Icon(
                        pending || rejected
                            ? Icons.lock_outline_rounded
                            : Icons.chevron_right_rounded,
                        size: 22,
                        color: AppColors.textDark.withOpacity(dimmed ? 0.35 : 0.6),
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
        ),
      ),
    );

    if (dimmed) {
      card = Opacity(opacity: pending ? 0.88 : 0.82, child: card);
    }

    return Padding(
      padding: const EdgeInsets.only(bottom: 14),
      child: card,
    );
  }
}
