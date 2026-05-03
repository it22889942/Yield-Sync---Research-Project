import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../../utils/app_colors.dart';
import '../../services/app_routes.dart';
import '../widgets/app_shell.dart';
import '../../services/labor_api.dart';
import '../../services/recommendation_api.dart';

class LaborListScreen extends StatefulWidget {
  const LaborListScreen({super.key});

  @override
  State<LaborListScreen> createState() => _LaborListScreenState();
}

class _LaborListScreenState extends State<LaborListScreen> {
  final _searchCtrl = TextEditingController();
  final ScrollController _scrollController = ScrollController();
  late LaborSearchArgs _args;

  bool _showScrollToTop = false;

  static const double _scrollToTopThreshold = 160;

  List<String> _locations = const ["Kurunegala"];
  List<String> _skills = const [];
  String _selectedLocation = "Kurunegala";
  String? _selectedSkill;
  bool _loadingMeta = false;
  bool _metaInit = false;

  /// Only apply [ModalRoute] arguments once. `didChangeDependencies` runs again
  /// when returning from labour details; re-applying null args was clearing search/results.
  bool _routeArgsApplied = false;

  /// When true: recommendation + semantic search. When false: normal keyword/name search.
  bool _useSemanticSearch = true;

  bool _loading = false;
  String? _error;
  List<LaborWorker> _items = [];
  String? _primaryLocationFromQuery;

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    if (!_routeArgsApplied) {
      _routeArgsApplied = true;
      final a = ModalRoute.of(context)?.settings.arguments;
      _args = (a is LaborSearchArgs)
          ? a
          : LaborSearchArgs(query: "", location: "Kurunegala", skill: null);

      _searchCtrl.text = _args.query;
      _selectedLocation = _args.location;
      final skillArg = _args.skill?.trim();
      _selectedSkill =
          (skillArg != null && skillArg.toLowerCase() == "unknown")
              ? null
              : _args.skill;
      if (!_metaInit) {
        _metaInit = true;
        _loadMeta();
      }
      _fetch();
    }
  }

  Future<void> _loadMeta() async {
    setState(() => _loadingMeta = true);
    try {
      final locs = await LaborApi.getLocations();
      final skills = await LaborApi.getSkills();
      if (!mounted) return;
      setState(() {
        _locations = locs.isEmpty ? const ["Kurunegala"] : locs;
        _skills = skills
            .where((s) => s.trim().toLowerCase() != "unknown")
            .toList();
        if (!_locations.contains(_selectedLocation)) {
          _selectedLocation = _locations.first;
        }
      });
    } catch (_) {
      // keep fallbacks
    } finally {
      if (mounted) setState(() => _loadingMeta = false);
    }
  }

  void _commitAndSearch() => _fetch();

  @override
  void initState() {
    super.initState();
    _scrollController.addListener(_onLaborListScroll);
  }

  void _onLaborListScroll() {
    if (!_scrollController.hasClients) return;
    final show = _scrollController.offset > _scrollToTopThreshold;
    if (show != _showScrollToTop && mounted) {
      setState(() => _showScrollToTop = show);
    }
  }

  @override
  void dispose() {
    _scrollController.removeListener(_onLaborListScroll);
    _scrollController.dispose();
    _searchCtrl.dispose();
    super.dispose();
  }

  Future<void> _fetch() async {
    setState(() {
      _loading = true;
      _error = null;
      _args = LaborSearchArgs(
        query: _searchCtrl.text.trim(),
        location: _selectedLocation,
        skill: _selectedSkill,
      );
    });

    try {
      final q = _searchCtrl.text.trim();
      _primaryLocationFromQuery = _extractLocationFromQuery(q);

      List<LaborWorker> list;

      if (_useSemanticSearch) {
        // Recommendation + semantic search (current behavior)
        final parts = <String>[];
        if (q.isNotEmpty) parts.add(q);
        if (_args.skill != null && _args.skill!.trim().isNotEmpty) {
          parts.add(_args.skill!.trim());
        }
        if (_args.location.trim().isNotEmpty) {
          parts.add('in ${_args.location.trim()}');
        }
        final composedQuery = parts.join(' ').trim();

        final recs = await RecommendationApi.recommendLabour(
          query: composedQuery.isEmpty ? _args.location : composedQuery,
          topK: 40,
        );
        list = recs.map((e) => LaborWorker.fromJson(e)).toList();
      } else {
        // Normal keyword/name search
        final map = await LaborApi.search(
          query: q,
          location: _args.location,
          skill: _args.skill,
          topK: 40,
        );
        final rawItems = map['items'] as List<dynamic>? ?? [];
        list = rawItems
            .map((e) =>
                LaborWorker.fromJson(Map<String, dynamic>.from(e as Map)))
            .toList();
      }

      if (!mounted) return;
      setState(() => _items = list);
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  String? _extractLocationFromQuery(String raw) {
    final q = raw.trim();
    if (q.isEmpty) return null;
    final exp = RegExp(r'\bin\s+([A-Za-z\s]+)$', caseSensitive: false);
    final m = exp.firstMatch(q);
    if (m == null) return null;
    final loc = m.group(1)?.trim();
    if (loc == null || loc.isEmpty) return null;
    return loc;
  }

  @override
  Widget build(BuildContext context) {
    final list = _items;
    final primaryKey = _primaryLocationFromQuery?.toLowerCase().trim();

    List<LaborWorker> primary = list;
    List<LaborWorker> others = const [];

    if (primaryKey != null && primaryKey.isNotEmpty) {
      primary = list
          .where((w) =>
              w.location.toLowerCase().contains(primaryKey))
          .toList();
      others = list
          .where((w) =>
              !w.location.toLowerCase().contains(primaryKey))
          .toList();
    }

    return AppShell(
      currentIndex: 0,
      child: SizedBox.expand(
        child: Stack(
          clipBehavior: Clip.none,
          children: [
            Positioned.fill(
              child: RefreshIndicator(
                onRefresh: _fetch,
                child: CustomScrollView(
                  controller: _scrollController,
                  physics: const AlwaysScrollableScrollPhysics(
                    parent: BouncingScrollPhysics(),
                  ),
                  slivers: [
                    SliverToBoxAdapter(child: _heroHeader()),
                    SliverPadding(
                      padding: const EdgeInsets.fromLTRB(16, 12, 16, 24),
                      sliver: SliverList(
                        delegate: SliverChildListDelegate(
                          _laborListResultWidgets(
                              context, list, primary, others),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),
            if (_showScrollToTop)
              Positioned(
                right: 14,
                bottom: 14,
                child: Material(
                  elevation: 6,
                  shadowColor: Colors.black26,
                  shape: const CircleBorder(),
                  color: AppColors.primary,
                  clipBehavior: Clip.antiAlias,
                  child: InkWell(
                    customBorder: const CircleBorder(),
                    onTap: () {
                      _scrollController.animateTo(
                        0,
                        duration: const Duration(milliseconds: 420),
                        curve: Curves.easeOutCubic,
                      );
                    },
                    child: const Padding(
                      padding: EdgeInsets.all(11),
                      child: Icon(
                        Icons.keyboard_arrow_up_rounded,
                        color: AppColors.darkGreen,
                        size: 26,
                      ),
                    ),
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }

  List<Widget> _laborListResultWidgets(
    BuildContext context,
    List<LaborWorker> list,
    List<LaborWorker> primary,
    List<LaborWorker> others,
  ) {
    if (_loading) {
      return [
        _skeletonCard(),
        const SizedBox(height: 12),
        _skeletonCard(),
        const SizedBox(height: 12),
        _skeletonCard(),
      ];
    }
    if (_error != null) {
      return [_errorBox(_error!, onRetry: _fetch)];
    }
    if (list.isEmpty) {
      return [
        _emptyBox(
          title: "No workers found",
          subtitle:
              "Try another keyword, location, or skill filter above.",
          onRetry: _fetch,
        ),
      ];
    }

    final widgets = <Widget>[
      ...primary.map(
        (w) => _WorkerCardModern(
          worker: w,
          onTap: () {
            Navigator.pushNamed(
              context,
              AppRoutes.laborDetails,
              arguments: w,
            );
          },
        ),
      ),
    ];

    if (primary.isNotEmpty && others.isNotEmpty) {
      widgets.addAll([
        const SizedBox(height: 18),
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
          decoration: BoxDecoration(
            color: AppColors.primary.withOpacity(0.08),
            borderRadius: BorderRadius.circular(999),
            border: Border.all(
              color: AppColors.primary.withOpacity(0.45),
              width: 1.2,
            ),
          ),
          child: Row(
            children: [
              const Expanded(
                child: Divider(
                  color: AppColors.border,
                  thickness: 1,
                ),
              ),
              const SizedBox(width: 10),
              const Icon(
                Icons.location_city_rounded,
                size: 16,
                color: AppColors.darkGreen,
              ),
              const SizedBox(width: 6),
              Text(
                "Other locations",
                style: GoogleFonts.poppins(
                  fontWeight: FontWeight.w700,
                  color: AppColors.textDark,
                  fontSize: 12.5,
                ),
              ),
              const SizedBox(width: 10),
              const Expanded(
                child: Divider(
                  color: AppColors.border,
                  thickness: 1,
                ),
              ),
            ],
          ),
        ),
        const SizedBox(height: 10),
      ]);
    }

    widgets.addAll(
      others.map(
        (w) => _WorkerCardModern(
          worker: w,
          onTap: () {
            Navigator.pushNamed(
              context,
              AppRoutes.laborDetails,
              arguments: w,
            );
          },
        ),
      ),
    );

    return widgets;
  }

  // ================= HERO =================
  Widget _heroHeader() {
    final skillLabel = (_selectedSkill == null || _selectedSkill!.trim().isEmpty)
        ? "Any skill"
        : _selectedSkill!.trim();

    return Container(
      width: double.infinity,
      decoration: const BoxDecoration(
        gradient: AppColors.heroGradient,
        borderRadius: BorderRadius.only(
          bottomLeft: Radius.circular(28),
          bottomRight: Radius.circular(28),
        ),
      ),
      clipBehavior: Clip.antiAlias,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 12, 16, 0),
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
                        "Hire Workers",
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
                      onTap: () {
                        _loadMeta();
                        _fetch();
                      },
                    ),
                  ],
                ),
                const SizedBox(height: 6),
                Align(
                  alignment: Alignment.centerLeft,
                  child: Text(
                    "Find Skilled Workers",
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
                Align(
                  alignment: Alignment.centerLeft,
                  child: Text(
                    "$skillLabel • $_selectedLocation",
                    maxLines: 2,
                    overflow: TextOverflow.ellipsis,
                    style: GoogleFonts.inter(
                      color: Colors.white.withOpacity(0.82),
                      fontWeight: FontWeight.w500,
                      fontSize: 12.5,
                      height: 1.35,
                    ),
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 8),
          Image.asset(
            'assets/images/farmerhead.png',
            width: double.infinity,
            fit: BoxFit.fitWidth,
            alignment: Alignment.topCenter,
            gaplessPlayback: true,
          ),
          const SizedBox(height: 8),
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 0, 16, 12),
            child: Container(
              padding: const EdgeInsets.fromLTRB(11, 9, 11, 10),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(16),
                border: Border.all(color: AppColors.border),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withOpacity(0.05),
                    blurRadius: 12,
                    offset: const Offset(0, 6),
                  ),
                ],
              ),
              child: Theme(
                data: Theme.of(context).copyWith(
                  visualDensity: VisualDensity.compact,
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    Row(
                      crossAxisAlignment: CrossAxisAlignment.center,
                      children: [
                        Text(
                          "Search & filter",
                          style: GoogleFonts.poppins(
                            fontWeight: FontWeight.w700,
                            fontSize: 13.5,
                            color: AppColors.textDark,
                          ),
                        ),
                        const Spacer(),
                        Text(
                          _useSemanticSearch ? "Smart" : "Name",
                          style: GoogleFonts.inter(
                            fontSize: 11.5,
                            color: AppColors.textDark.withOpacity(0.72),
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                        const SizedBox(width: 2),
                        Transform.scale(
                          scale: 0.82,
                          alignment: Alignment.center,
                          child: Switch(
                            value: _useSemanticSearch,
                            onChanged: (v) {
                              setState(() => _useSemanticSearch = v);
                              _fetch();
                            },
                            activeColor: AppColors.primary,
                            materialTapTargetSize:
                                MaterialTapTargetSize.shrinkWrap,
                          ),
                        ),
                        if (_loadingMeta || _loading) ...[
                          const SizedBox(width: 4),
                          const SizedBox(
                            width: 14,
                            height: 14,
                            child: CircularProgressIndicator(strokeWidth: 2),
                          ),
                        ],
                      ],
                    ),
                    const SizedBox(height: 6),
                    _laborListSearchField(),
                    const SizedBox(height: 14),
                    _laborListLocationField(),
                    const SizedBox(height: 14),
                    Row(
                      crossAxisAlignment: CrossAxisAlignment.center,
                      children: [
                        Text(
                          "Skill",
                          style: GoogleFonts.poppins(
                            fontWeight: FontWeight.w700,
                            fontSize: 11.5,
                            color: AppColors.textDark.withOpacity(0.72),
                          ),
                        ),
                        const SizedBox(width: 6),
                        Expanded(
                          child: SizedBox(
                            height: 32,
                            child: _laborListSkillChipsHorizontal(),
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _laborListSearchField() {
    return TextField(
      controller: _searchCtrl,
      textInputAction: TextInputAction.search,
      style: GoogleFonts.inter(
        fontSize: 14,
        height: 1.3,
        fontWeight: FontWeight.w500,
        color: AppColors.textDark,
      ),
      onSubmitted: (_) => _commitAndSearch(),
      decoration: InputDecoration(
        isDense: true,
        hintText: "Name or skill…",
        hintStyle: GoogleFonts.inter(
          fontSize: 13,
          color: AppColors.textDark.withOpacity(0.45),
          fontWeight: FontWeight.w500,
        ),
        prefixIcon: const Icon(Icons.search_rounded, size: 20),
        prefixIconConstraints:
            const BoxConstraints(minWidth: 40, minHeight: 36),
        suffixIcon: _searchCtrl.text.trim().isEmpty
            ? null
            : IconButton(
                padding: EdgeInsets.zero,
                constraints: const BoxConstraints(
                  minWidth: 36,
                  minHeight: 36,
                ),
                onPressed: () {
                  setState(() => _searchCtrl.clear());
                  _commitAndSearch();
                },
                icon: const Icon(Icons.close_rounded, size: 20),
              ),
        contentPadding:
            const EdgeInsets.symmetric(horizontal: 10, vertical: 10),
        filled: true,
        fillColor: AppColors.surface,
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(12),
          borderSide: const BorderSide(color: AppColors.border),
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(12),
          borderSide: const BorderSide(color: AppColors.border),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(12),
          borderSide:
              const BorderSide(color: AppColors.primary, width: 1.4),
        ),
      ),
      onChanged: (_) => setState(() {}),
    );
  }

  Widget _laborListLocationField() {
    return DropdownButtonFormField<String>(
      isDense: true,
      value: _locations.contains(_selectedLocation)
          ? _selectedLocation
          : _locations.first,
      items: _locations
          .map(
            (e) => DropdownMenuItem(
              value: e,
              child: Text(
                e,
                style: GoogleFonts.inter(
                  fontSize: 14,
                  fontWeight: FontWeight.w500,
                ),
              ),
            ),
          )
          .toList(),
      onChanged: (v) {
        if (v == null) return;
        setState(() => _selectedLocation = v);
        _commitAndSearch();
      },
      style: GoogleFonts.inter(
        fontSize: 14,
        fontWeight: FontWeight.w600,
        color: AppColors.textDark,
      ),
      decoration: InputDecoration(
        labelText: "Your Location",
        labelStyle: GoogleFonts.inter(
          fontSize: 12.5,
          fontWeight: FontWeight.w600,
        ),
        prefixIcon: const Icon(Icons.location_on_rounded, size: 20),
        prefixIconConstraints:
            const BoxConstraints(minWidth: 40, minHeight: 36),
        contentPadding:
            const EdgeInsets.symmetric(horizontal: 8, vertical: 8),
        filled: true,
        fillColor: AppColors.surface,
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(12),
          borderSide: const BorderSide(color: AppColors.border),
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(12),
          borderSide: const BorderSide(color: AppColors.border),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(12),
          borderSide:
              const BorderSide(color: AppColors.primary, width: 1.4),
        ),
      ),
    );
  }

  Widget _laborListSkillChipsHorizontal() {
    final allSkills = _skills
        .where((s) =>
            s.trim().isNotEmpty &&
            s.trim().toLowerCase() != "unknown")
        .toList();

    if (_loadingMeta && allSkills.isEmpty) {
      return ListView.separated(
        scrollDirection: Axis.horizontal,
        itemCount: 6,
        separatorBuilder: (_, __) => const SizedBox(width: 6),
        itemBuilder: (_, __) => Container(
          width: 72,
          height: 28,
          decoration: BoxDecoration(
            color: AppColors.surface,
            borderRadius: BorderRadius.circular(999),
            border: Border.all(color: AppColors.border),
          ),
        ),
      );
    }

    return ListView(
      scrollDirection: Axis.horizontal,
      children: [
        _skillPillChip(
          label: "Any skill",
          selected: _selectedSkill == null,
          onTap: () {
            setState(() => _selectedSkill = null);
            _commitAndSearch();
          },
          icon: Icons.all_inclusive_rounded,
        ),
        ...allSkills.take(36).map((s) {
          final selected = _selectedSkill == s;
          return Padding(
            padding: const EdgeInsets.only(left: 6),
            child: _skillPillChip(
              label: s,
              selected: selected,
              onTap: () {
                setState(() => _selectedSkill = selected ? null : s);
                _commitAndSearch();
              },
              icon: Icons.badge_rounded,
            ),
          );
        }),
      ],
    );
  }

  Widget _skillPillChip({
    required String label,
    required bool selected,
    required VoidCallback onTap,
    required IconData icon,
  }) {
    return InkWell(
      borderRadius: BorderRadius.circular(999),
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 5),
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
              size: 13,
              color: selected
                  ? AppColors.darkGreen
                  : AppColors.textDark.withOpacity(0.55),
            ),
            const SizedBox(width: 4),
            ConstrainedBox(
              constraints: const BoxConstraints(maxWidth: 120),
              child: Text(
                label,
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
                style: GoogleFonts.inter(
                  fontWeight: FontWeight.w600,
                  fontSize: 11.5,
                  height: 1.15,
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

  // ================= Empty / Error / Skeleton =================
  Widget _emptyBox({
    required String title,
    required String subtitle,
    required VoidCallback onRetry,
  }) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: AppColors.border),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            title,
            style: GoogleFonts.poppins(
              fontWeight: FontWeight.w700,
              fontSize: 17,
              color: AppColors.textDark,
            ),
          ),
          const SizedBox(height: 6),
          Text(
            subtitle,
            style: GoogleFonts.inter(
              fontWeight: FontWeight.w500,
              fontSize: 13.2,
              height: 1.4,
              color: AppColors.textDark.withOpacity(0.65),
            ),
          ),
          const SizedBox(height: 12),
          SizedBox(
            height: 46,
            child: ElevatedButton.icon(
              onPressed: onRetry,
              icon: const Icon(Icons.refresh_rounded),
              label: Text(
                "Retry",
                style: GoogleFonts.poppins(
                  fontWeight: FontWeight.w700,
                  letterSpacing: 0.15,
                ),
              ),
              style: ElevatedButton.styleFrom(
                backgroundColor: AppColors.primary,
                foregroundColor: AppColors.darkGreen,
                elevation: 0,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(14),
                ),
              ),
            ),
          )
        ],
      ),
    );
  }

  Widget _errorBox(String message, {required VoidCallback onRetry}) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.red.withOpacity(0.06),
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: Colors.red.withOpacity(0.2)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            "Error loading workers",
            style: GoogleFonts.poppins(
              fontWeight: FontWeight.w700,
              fontSize: 17,
              color: Colors.red,
            ),
          ),
          const SizedBox(height: 6),
          Text(
            message,
            style: GoogleFonts.inter(
              fontWeight: FontWeight.w500,
              fontSize: 13.2,
              height: 1.35,
              color: Colors.red.withOpacity(0.88),
            ),
          ),
          const SizedBox(height: 12),
          SizedBox(
            height: 46,
            child: ElevatedButton.icon(
              onPressed: onRetry,
              icon: const Icon(Icons.refresh_rounded),
              label: Text(
                "Retry",
                style: GoogleFonts.poppins(
                  fontWeight: FontWeight.w700,
                  letterSpacing: 0.15,
                ),
              ),
              style: ElevatedButton.styleFrom(
                backgroundColor: AppColors.primary,
                foregroundColor: AppColors.darkGreen,
                elevation: 0,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(14),
                ),
              ),
            ),
          )
        ],
      ),
    );
  }

  Widget _skeletonCard() {
    Widget bar({double w = double.infinity, double h = 12}) => Container(
          width: w,
          height: h,
          decoration: BoxDecoration(
            color: AppColors.surface,
            borderRadius: BorderRadius.circular(10),
            border: Border.all(color: AppColors.border),
          ),
        );

    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: AppColors.border),
      ),
      child: Row(
        children: [
          Container(
            width: 56,
            height: 56,
            decoration: BoxDecoration(
              color: AppColors.surface,
              borderRadius: BorderRadius.circular(18),
              border: Border.all(color: AppColors.border),
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                bar(w: 160, h: 14),
                const SizedBox(height: 10),
                bar(w: 220),
                const SizedBox(height: 10),
                Row(
                  children: [
                    bar(w: 80, h: 26),
                    const SizedBox(width: 10),
                    bar(w: 90, h: 26),
                  ],
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

// ================= Modern Worker Card =================
class _WorkerCardModern extends StatelessWidget {
  final LaborWorker worker;
  final VoidCallback onTap;

  const _WorkerCardModern({required this.worker, required this.onTap});

  String _initials(String name) {
    final n = name.trim();
    if (n.isEmpty) return "W";
    final parts = n.split(" ").where((e) => e.trim().isNotEmpty).toList();
    if (parts.length == 1) return parts.first[0].toUpperCase();
    return "${parts[0][0]}${parts[1][0]}".toUpperCase();
  }

  Color _scoreColor(double s) {
    if (s >= 0.75) return const Color(0xFF2FA36B);
    if (s >= 0.5) return const Color(0xFFC9A80B);
    return const Color(0xFFE05C5C);
  }

  @override
  Widget build(BuildContext context) {
    final score = worker.score;
    final scorePct =
        (score <= 1.0) ? (score * 100) : score; // supports 0-1 or 0-100
    final scColor = _scoreColor(score <= 1.0 ? score : (score / 100.0));

    return Container(
      margin: const EdgeInsets.only(bottom: 12),
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
              children: [
                Row(
                  children: [
                    CircleAvatar(
                      radius: 24,
                      backgroundColor: AppColors.primary.withOpacity(0.16),
                      child: Text(
                        _initials(worker.name),
                        style: GoogleFonts.poppins(
                          fontWeight: FontWeight.w800,
                          fontSize: 15,
                          color: AppColors.textDark,
                        ),
                      ),
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            worker.name,
                            maxLines: 1,
                            overflow: TextOverflow.ellipsis,
                            style: GoogleFonts.poppins(
                              fontWeight: FontWeight.w700,
                              fontSize: 15.5,
                              letterSpacing: -0.15,
                              color: AppColors.textDark,
                            ),
                          ),
                          const SizedBox(height: 4),
                          Row(
                            children: [
                              const Icon(Icons.location_on_rounded,
                                  size: 16, color: AppColors.darkGreen),
                              const SizedBox(width: 4),
                              Expanded(
                                child: Text(
                                  worker.location,
                                  maxLines: 1,
                                  overflow: TextOverflow.ellipsis,
                                  style: GoogleFonts.inter(
                                    fontWeight: FontWeight.w500,
                                    fontSize: 12.8,
                                    color:
                                        AppColors.textDark.withOpacity(0.68),
                                  ),
                                ),
                              ),
                            ],
                          ),
                        ],
                      ),
                    ),
                    Column(
                      crossAxisAlignment: CrossAxisAlignment.end,
                      children: [
                        Text(
                          "LKR ${worker.hourlyRate.toStringAsFixed(0)}/hr",
                          style: GoogleFonts.poppins(
                            fontWeight: FontWeight.w700,
                            fontSize: 14,
                            color: AppColors.darkGreen,
                          ),
                        ),
                        const SizedBox(height: 6),
                        Container(
                          padding: const EdgeInsets.symmetric(
                              horizontal: 10, vertical: 6),
                          decoration: BoxDecoration(
                            color: scColor.withOpacity(0.10),
                            borderRadius: BorderRadius.circular(999),
                            border:
                                Border.all(color: scColor.withOpacity(0.25)),
                          ),
                          child: Text(
                            "Match ${scorePct.toStringAsFixed(0)}%",
                            style: GoogleFonts.inter(
                              fontWeight: FontWeight.w700,
                              color: scColor,
                              fontSize: 11.5,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
                const SizedBox(height: 12),
                Row(
                  children: [
                    ...List.generate(
                      5,
                      (i) => Icon(
                        i < worker.rating.round()
                            ? Icons.star_rounded
                            : Icons.star_border_rounded,
                        size: 18,
                        color: const Color(0xFFF5B400),
                      ),
                    ),
                    const SizedBox(width: 8),
                    Text(
                      worker.rating.toStringAsFixed(1),
                      style: GoogleFonts.inter(
                        fontWeight: FontWeight.w700,
                        fontSize: 13.5,
                        color: AppColors.textDark,
                      ),
                    ),
                    const Spacer(),
                    Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 10, vertical: 6),
                      decoration: BoxDecoration(
                        color: AppColors.primary.withOpacity(0.10),
                        borderRadius: BorderRadius.circular(999),
                        border: Border.all(color: AppColors.border),
                      ),
                      child: Text(
                        worker.labourType,
                        style: GoogleFonts.poppins(
                          fontWeight: FontWeight.w600,
                          color: AppColors.darkGreen,
                          fontSize: 12,
                        ),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 12),
                Wrap(
                  spacing: 8,
                  runSpacing: 8,
                  children: [
                    _chip("Skill: ${worker.skillLevel}"),
                    _chip("Crop: ${worker.cropType}"),
                    _chip("Exp: ${worker.experienceYears}y"),
                    _chip("Jobs: ${worker.jobsCompleted}"),
                    if (worker.availableDay.isNotEmpty ||
                        worker.availableTime.isNotEmpty)
                      _chip(
                          "Avail: ${worker.availableDay} ${worker.availableTime}"
                              .trim()),
                  ],
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _chip(String s) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 7),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(999),
        border: Border.all(color: AppColors.border),
      ),
      child: Text(
        s,
        style: GoogleFonts.inter(
          fontWeight: FontWeight.w500,
          color: AppColors.textDark.withOpacity(0.78),
          fontSize: 12,
          height: 1.25,
        ),
      ),
    );
  }
}

// ===== Model from backend JSON (unchanged) =====
class LaborWorker {
  final String id;
  final String name;
  final String location;
  final String labourType;
  final String skillLevel;
  final double hourlyRate;
  final double rating;
  final int experienceYears;
  final int jobsCompleted;
  final String season;
  final String cropType;
  final String availableDay;
  final String availableTime;
  final double score;

  LaborWorker({
    required this.id,
    required this.name,
    required this.location,
    required this.labourType,
    required this.skillLevel,
    required this.hourlyRate,
    required this.rating,
    required this.experienceYears,
    required this.jobsCompleted,
    required this.season,
    required this.cropType,
    required this.availableDay,
    required this.availableTime,
    required this.score,
  });

  factory LaborWorker.fromJson(Map m) {
    double _d(v) =>
        (v is num) ? v.toDouble() : double.tryParse(v.toString()) ?? 0;
    int _i(v) => (v is num) ? v.toInt() : int.tryParse(v.toString()) ?? 0;

    return LaborWorker(
      // Support both /api/labour/search (lowercase keys)
      // and /api/recommend/recommend (DataFrame column-style keys).
      id: (m["id"] ?? m["Labour_ID"] ?? "").toString(),
      name: (m["name"] ?? m["Name"] ?? "").toString(),
      location: (m["location"] ?? m["Location"] ?? "").toString(),
      labourType: (m["labour_type"] ?? m["Labour_Type"] ?? "").toString(),
      skillLevel:
          (m["skill_level"] ?? m["Skill_Level"] ?? "").toString(),
      hourlyRate: _d(m["hourly_rate"] ?? m["Hourly_Rate"]),
      rating: _d(m["rating"] ?? m["Rating"]),
      experienceYears:
          _i(m["experience_years"] ?? m["Experience_Years"]),
      jobsCompleted:
          _i(m["jobs_completed"] ?? m["Jobs_Completed"]),
      season: (m["season"] ?? m["Season"] ?? "").toString(),
      cropType: (m["crop_type"] ?? m["Crop_Type"] ?? "").toString(),
      availableDay:
          (m["available_day"] ?? m["Available_Day"] ?? "").toString(),
      availableTime:
          (m["available_time"] ?? m["Available_Time"] ?? "").toString(),
      score: _d(m["score"]),
    );
  }
}
