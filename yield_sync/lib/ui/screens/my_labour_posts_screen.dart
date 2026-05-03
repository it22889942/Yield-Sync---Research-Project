import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/material.dart';

import '../../utils/app_colors.dart';
import '../../services/app_routes.dart';
import '../../services/labour_equipment_post_service.dart';
import '../widgets/app_shell.dart';
import 'labor_list_screen.dart';

class MyLabourPostsScreen extends StatelessWidget {
  const MyLabourPostsScreen({super.key});

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
                  stream: LabourEquipmentPostService.myLabourPostsStream(),
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
                            Icon(Icons.groups_rounded,
                                size: 64, color: AppColors.textDark.withOpacity(0.3)),
                            const SizedBox(height: 16),
                            Text(
                              "No labour ads yet",
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
                        final worker = LaborWorker.fromJson({
                          ...m,
                          "Labour_ID": d.id,
                          "id": d.id,
                        });
                        final mod =
                            LabourEquipmentPostService.moderationStateFromData(m);
                        return _PostCard(
                          name: worker.name.isEmpty ? "Labour ${d.id}" : worker.name,
                          labourType: worker.labourType,
                          location: worker.location,
                          hourlyRate: worker.hourlyRate,
                          moderation: mod,
                          onOpen: () => Navigator.pushNamed(
                            context,
                            AppRoutes.laborDetails,
                            arguments: worker,
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
                  "Your labour ads",
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
  final String name;
  final String labourType;
  final String location;
  final double hourlyRate;
  final PostModerationState moderation;
  final VoidCallback onOpen;

  const _PostCard({
    required this.name,
    required this.labourType,
    required this.location,
    required this.hourlyRate,
    required this.moderation,
    required this.onOpen,
  });

  static const _iconBg = Color(0xFF15B77E);

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
                            : Icons.groups_rounded,
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
                              name,
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
                      if (labourType.isNotEmpty) ...[
                        const SizedBox(height: 4),
                        Text(
                          labourType,
                          style: TextStyle(
                            fontWeight: FontWeight.w700,
                            color: AppColors.textDark.withOpacity(dimmed ? 0.4 : 0.65),
                            fontSize: 13,
                          ),
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                        ),
                      ],
                      if (location.isNotEmpty) ...[
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
                                location,
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
                        "LKR ${hourlyRate.toStringAsFixed(0)}/hr",
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
