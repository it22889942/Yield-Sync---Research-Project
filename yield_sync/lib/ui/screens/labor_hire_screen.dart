import 'package:flutter/material.dart';
import '../../utils/app_colors.dart';
import '../../services/app_routes.dart';
import '../widgets/app_shell.dart';
import '../../services/labor_api.dart';

class LaborHireScreen extends StatefulWidget {
  const LaborHireScreen({super.key});

  @override
  State<LaborHireScreen> createState() => _LaborHireScreenState();
}

class _LaborHireScreenState extends State<LaborHireScreen> {
  final _searchCtrl = TextEditingController();

  List<String> _locations = const ["Kurunegala"];
  List<String> _skills = const [];

  String _location = "Kurunegala";
  String? _skill;

  bool _loadingMeta = false;

  @override
  void initState() {
    super.initState();
    _loadMeta();
  }

  Future<void> _loadMeta() async {
    setState(() => _loadingMeta = true);
    try {
      final locs = await LaborApi.getLocations();
      final skills = await LaborApi.getSkills();
      if (!mounted) return;
      setState(() {
        _locations = locs.isEmpty ? const ["Kurunegala"] : locs;
        _skills = skills;
        if (!_locations.contains(_location)) _location = _locations.first;
      });
    } catch (_) {
      // keep fallback if backend down
    } finally {
      if (mounted) setState(() => _loadingMeta = false);
    }
  }

  @override
  void dispose() {
    _searchCtrl.dispose();
    super.dispose();
  }

  void _goSearch() {
    Navigator.pushNamed(
      context,
      AppRoutes.laborList,
      arguments: LaborSearchArgs(
        query: _searchCtrl.text.trim(),
        location: _location,
        skill: _skill,
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return AppShell(
      currentIndex: 0,
      child: Column(
        children: [
          _heroHeader(),
          Expanded(
            child: Stack(
              children: [
                ListView(
                  padding: const EdgeInsets.fromLTRB(16, 12, 16, 120),
                  children: [
                    _whiteCard(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Row(
                            children: [
                              const Text(
                                "Search Workers",
                                style: TextStyle(
                                  fontWeight: FontWeight.w900,
                                  fontSize: 16,
                                  color: AppColors.textDark,
                                ),
                              ),
                              const Spacer(),
                              if (_loadingMeta)
                                const SizedBox(
                                  height: 16,
                                  width: 16,
                                  child:
                                      CircularProgressIndicator(strokeWidth: 2),
                                ),
                            ],
                          ),
                          const SizedBox(height: 12),

                          // Search input (modern)
                          _searchField(),

                          const SizedBox(height: 12),

                          // Location dropdown (clean)
                          _locationDropdown(),

                          const SizedBox(height: 14),

                          // Skills (modern chips)
                          _skillChips(),
                        ],
                      ),
                    ),

                    const SizedBox(height: 12),

                    // Friendly hint card
                    _hintCard(),
                  ],
                ),

                // Sticky CTA
                Positioned(
                  left: 16,
                  right: 16,
                  bottom: 12,
                  child: SafeArea(
                    top: false,
                    child: _stickySearchBar(),
                  ),
                )
              ],
            ),
          ),
        ],
      ),
    );
  }

  // ================= HERO =================
  Widget _heroHeader() {
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
      child: Column(
        children: [
          Row(
            children: [
              _roundIconBtn(
                icon: Icons.arrow_back_ios_new_rounded,
                onTap: () => Navigator.pop(context),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: Text(
                  "Hire Workers",
                  style: TextStyle(
                    color: Colors.white.withOpacity(0.95),
                    fontWeight: FontWeight.w900,
                    fontSize: 16,
                  ),
                ),
              ),
              _roundIconBtn(
                icon: Icons.refresh_rounded,
                onTap: _loadMeta,
              ),
            ],
          ),
          const SizedBox(height: 10),
          Align(
            alignment: Alignment.centerLeft,
            child: Text(
              "Find Skilled Workers",
              style: TextStyle(
                color: Colors.white.withOpacity(0.97),
                fontWeight: FontWeight.w900,
                fontSize: 24,
              ),
            ),
          ),
          const SizedBox(height: 6),
          Align(
            alignment: Alignment.centerLeft,
            child: Text(
              "Search by name or skill • filter by district • fast booking",
              style: TextStyle(
                color: Colors.white.withOpacity(0.75),
                fontWeight: FontWeight.w600,
                fontSize: 12.8,
              ),
            ),
          ),
          const SizedBox(height: 12),
          Row(
            children: [
              _headerChip(
                icon: Icons.location_on_rounded,
                label: _location,
              ),
              const SizedBox(width: 10),
              _headerChip(
                icon: Icons.badge_rounded,
                label:
                    _skills.isEmpty ? "Skills —" : "${_skills.length} skills",
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _headerChip({required IconData icon, required String label}) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.14),
        borderRadius: BorderRadius.circular(999),
        border: Border.all(color: Colors.white.withOpacity(0.20)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 16, color: Colors.white.withOpacity(0.95)),
          const SizedBox(width: 8),
          Text(
            label,
            style: TextStyle(
              color: Colors.white.withOpacity(0.92),
              fontWeight: FontWeight.w800,
              fontSize: 12.3,
            ),
          ),
        ],
      ),
    );
  }

  Widget _roundIconBtn({required IconData icon, required VoidCallback onTap}) {
    return InkWell(
      borderRadius: BorderRadius.circular(999),
      onTap: onTap,
      child: CircleAvatar(
        radius: 18,
        backgroundColor: Colors.white.withOpacity(0.14),
        child: Icon(icon, color: Colors.white),
      ),
    );
  }

  // ================= FORM UI =================
  Widget _searchField() {
    return TextField(
      controller: _searchCtrl,
      textInputAction: TextInputAction.search,
      onSubmitted: (_) => _goSearch(),
      decoration: InputDecoration(
        hintText: "Search labor (name/skill) e.g. paddy harvesting",
        prefixIcon: const Icon(Icons.search_rounded),
        suffixIcon: _searchCtrl.text.trim().isEmpty
            ? null
            : IconButton(
                onPressed: () => setState(() => _searchCtrl.clear()),
                icon: const Icon(Icons.close_rounded),
              ),
        filled: true,
        fillColor: AppColors.surface,
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(16),
          borderSide: const BorderSide(color: AppColors.border),
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(16),
          borderSide: const BorderSide(color: AppColors.border),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(16),
          borderSide: const BorderSide(color: AppColors.primary, width: 1.6),
        ),
      ),
      onChanged: (_) => setState(() {}),
    );
  }

  Widget _locationDropdown() {
    return DropdownButtonFormField<String>(
      value: _location,
      items: _locations
          .map((e) => DropdownMenuItem(value: e, child: Text(e)))
          .toList(),
      onChanged: (v) => setState(() => _location = v!),
      decoration: InputDecoration(
        labelText: "Your location",
        prefixIcon: const Icon(Icons.location_on_rounded),
        filled: true,
        fillColor: AppColors.surface,
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(16),
          borderSide: const BorderSide(color: AppColors.border),
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(16),
          borderSide: const BorderSide(color: AppColors.border),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(16),
          borderSide: const BorderSide(color: AppColors.primary, width: 1.6),
        ),
      ),
    );
  }

  // Modern chip selector (better UX than dropdown)
  Widget _skillChips() {
    final allSkills = _skills.where((s) => s.trim().isNotEmpty).toList();

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          "Skill (optional)",
          style: TextStyle(
            fontWeight: FontWeight.w900,
            fontSize: 13,
            color: AppColors.textDark,
          ),
        ),
        const SizedBox(height: 10),
        if (_loadingMeta && allSkills.isEmpty)
          Wrap(
            spacing: 10,
            runSpacing: 10,
            children: List.generate(
              6,
              (_) => Container(
                height: 34,
                width: 96,
                decoration: BoxDecoration(
                  color: AppColors.surface,
                  borderRadius: BorderRadius.circular(999),
                  border: Border.all(color: AppColors.border),
                ),
              ),
            ),
          )
        else
          Wrap(
            spacing: 10,
            runSpacing: 10,
            children: [
              _pillChip(
                label: "Any skill",
                selected: _skill == null,
                onTap: () => setState(() => _skill = null),
                icon: Icons.all_inclusive_rounded,
              ),
              ...allSkills.take(18).map((s) {
                final selected = _skill == s;
                return _pillChip(
                  label: s,
                  selected: selected,
                  onTap: () => setState(() => _skill = selected ? null : s),
                  icon: Icons.badge_rounded,
                );
              }).toList(),
            ],
          ),
      ],
    );
  }

  Widget _pillChip({
    required String label,
    required bool selected,
    required VoidCallback onTap,
    required IconData icon,
  }) {
    return InkWell(
      borderRadius: BorderRadius.circular(999),
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 9),
        decoration: BoxDecoration(
          color: selected
              ? AppColors.primary.withOpacity(0.18)
              : AppColors.surface,
          borderRadius: BorderRadius.circular(999),
          border: Border.all(
            color: selected ? AppColors.primary : AppColors.border,
          ),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(
              icon,
              size: 16,
              color: selected
                  ? AppColors.darkGreen
                  : AppColors.textDark.withOpacity(0.55),
            ),
            const SizedBox(width: 8),
            ConstrainedBox(
              constraints: const BoxConstraints(maxWidth: 180),
              child: Text(
                label,
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
                style: TextStyle(
                  fontWeight: FontWeight.w900,
                  color: selected
                      ? AppColors.textDark
                      : AppColors.textDark.withOpacity(0.75),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  // ================= Sticky CTA =================
  Widget _stickySearchBar() {
    final hasQuery = _searchCtrl.text.trim().isNotEmpty;

    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: AppColors.border),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.10),
            blurRadius: 22,
            offset: const Offset(0, 14),
          ),
        ],
      ),
      child: Row(
        children: [
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  _skill == null ? "Searching all skills" : "Skill: $_skill",
                  style: const TextStyle(
                    fontWeight: FontWeight.w900,
                    color: AppColors.textDark,
                    fontSize: 13.5,
                  ),
                ),
                const SizedBox(height: 2),
                Text(
                  hasQuery
                      ? "Query: ${_searchCtrl.text.trim()}"
                      : "Add keyword (optional)",
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: TextStyle(
                    fontWeight: FontWeight.w700,
                    color: AppColors.textDark.withOpacity(0.55),
                    fontSize: 12.2,
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(width: 10),
          SizedBox(
            height: 50,
            child: ElevatedButton.icon(
              onPressed: _goSearch,
              icon: const Icon(Icons.search_rounded),
              label: const Text(
                "Search",
                style: TextStyle(fontWeight: FontWeight.w900),
              ),
              style: ElevatedButton.styleFrom(
                backgroundColor: AppColors.primary,
                foregroundColor: AppColors.darkGreen,
                elevation: 0,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(16),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  // ================= Hint card =================
  Widget _hintCard() {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: AppColors.primary.withOpacity(0.10),
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: AppColors.border),
      ),
      child: Row(
        children: [
          Container(
            width: 42,
            height: 42,
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(16),
              border: Border.all(color: AppColors.border),
            ),
            child:
                const Icon(Icons.lightbulb_rounded, color: AppColors.darkGreen),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Text(
              "Tip: Select a skill chip to filter quickly. You can also leave it as “Any skill”.",
              style: TextStyle(
                fontWeight: FontWeight.w800,
                color: AppColors.textDark.withOpacity(0.70),
              ),
            ),
          ),
        ],
      ),
    );
  }

  // ================= Base card =================
  Widget _whiteCard({required Widget child}) {
    return Container(
      padding: const EdgeInsets.fromLTRB(16, 16, 16, 16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(22),
        border: Border.all(color: AppColors.border),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.06),
            blurRadius: 22,
            offset: const Offset(0, 14),
          ),
        ],
      ),
      child: child,
    );
  }
}

class LaborSearchArgs {
  final String query;
  final String location;
  final String? skill;

  LaborSearchArgs({
    required this.query,
    required this.location,
    this.skill,
  });
}
