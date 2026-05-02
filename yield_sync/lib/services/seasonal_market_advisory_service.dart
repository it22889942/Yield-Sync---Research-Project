class SeasonalMarketAdvisory {
  final int month;
  final int weekOfMonth;
  final String signal;
  final String title;
  final String detail;

  const SeasonalMarketAdvisory({
    required this.month,
    required this.weekOfMonth,
    required this.signal,
    required this.title,
    required this.detail,
  });
}

class SeasonalMarketAdvisoryService {
  const SeasonalMarketAdvisoryService();

  static const List<String> _monthNames = <String>[
    '',
    'January',
    'February',
    'March',
    'April',
    'May',
    'June',
    'July',
    'August',
    'September',
    'October',
    'November',
    'December',
  ];

  static const List<SeasonalMarketAdvisory> _advisories =
      <SeasonalMarketAdvisory>[
    SeasonalMarketAdvisory(
      month: 1,
      weekOfMonth: 2,
      signal: 'HOLD',
      title: 'Prices may increase before harvest supply enters market.',
      detail: 'Wait and sell later.',
    ),
    SeasonalMarketAdvisory(
      month: 1,
      weekOfMonth: 3,
      signal: 'SELL',
      title: 'Price has increased.',
      detail: 'Sell now and gain profit.',
    ),
    SeasonalMarketAdvisory(
      month: 2,
      weekOfMonth: 3,
      signal: 'SELL',
      title: 'Harvest supply is increasing and prices may start to drop.',
      detail: 'Sell now.',
    ),
    SeasonalMarketAdvisory(
      month: 3,
      weekOfMonth: 2,
      signal: 'HOLD',
      title: 'Prices may rise before Sinhala and Tamil New Year season.',
      detail: 'Wait until April.',
    ),
    SeasonalMarketAdvisory(
      month: 3,
      weekOfMonth: 3,
      signal: 'SELL',
      title: 'Festival buying has started.',
      detail: 'Good time to sell.',
    ),
    SeasonalMarketAdvisory(
      month: 4,
      weekOfMonth: 1,
      signal: 'HOLD',
      title: 'Festival week and prices are increasing.',
      detail: 'Wait a few days.',
    ),
    SeasonalMarketAdvisory(
      month: 4,
      weekOfMonth: 2,
      signal: 'SELL',
      title: 'Price reached peak level.',
      detail: 'Sell now.',
    ),
    SeasonalMarketAdvisory(
      month: 5,
      weekOfMonth: 2,
      signal: 'HOLD',
      title: 'Price is slightly stable.',
      detail: 'Wait one week.',
    ),
    SeasonalMarketAdvisory(
      month: 5,
      weekOfMonth: 3,
      signal: 'SELL',
      title: 'Supply increases after festival season and prices may decrease.',
      detail: 'Sell early.',
    ),
    SeasonalMarketAdvisory(
      month: 6,
      weekOfMonth: 2,
      signal: 'HOLD',
      title: 'Prices may slightly increase due to seasonal supply changes.',
      detail: 'Wait one week.',
    ),
    SeasonalMarketAdvisory(
      month: 6,
      weekOfMonth: 3,
      signal: 'SELL',
      title: 'Price increased as predicted.',
      detail: 'Sell now.',
    ),
    SeasonalMarketAdvisory(
      month: 7,
      weekOfMonth: 3,
      signal: 'SELL',
      title: 'Mid-year harvest increases supply and prices may fall.',
      detail: 'Sell now.',
    ),
    SeasonalMarketAdvisory(
      month: 8,
      weekOfMonth: 2,
      signal: 'HOLD',
      title: 'Prices are stable and may rise due to lower supply.',
      detail: 'Wait and monitor one week.',
    ),
    SeasonalMarketAdvisory(
      month: 8,
      weekOfMonth: 2,
      signal: 'SELL',
      title: 'Price increased slightly.',
      detail: 'Sell now.',
    ),
    SeasonalMarketAdvisory(
      month: 9,
      weekOfMonth: 4,
      signal: 'SELL',
      title: 'New harvest season begins and supply increases.',
      detail: 'Sell before price drops.',
    ),
    SeasonalMarketAdvisory(
      month: 10,
      weekOfMonth: 2,
      signal: 'HOLD',
      title: 'Prices may increase before Deepavali festival.',
      detail: 'Wait and sell later.',
    ),
    SeasonalMarketAdvisory(
      month: 10,
      weekOfMonth: 3,
      signal: 'SELL',
      title: 'Festival week with high price.',
      detail: 'Sell now.',
    ),
    SeasonalMarketAdvisory(
      month: 11,
      weekOfMonth: 3,
      signal: 'SELL',
      title: 'Festival season ends and supply increases.',
      detail: 'Prices may reduce, sell now.',
    ),
    SeasonalMarketAdvisory(
      month: 12,
      weekOfMonth: 1,
      signal: 'HOLD',
      title: 'Year-end demand may increase price.',
      detail: 'Wait one week.',
    ),
    SeasonalMarketAdvisory(
      month: 12,
      weekOfMonth: 2,
      signal: 'SELL',
      title: 'Price improved before holidays.',
      detail: 'Sell now.',
    ),
  ];

  int weekOfMonth(DateTime date) => ((date.day - 1) ~/ 7) + 1;

  String monthName(int month) => _monthNames[month];

  String seasonName(DateTime date) {
    final month = date.month;
    if (month >= 9 || month <= 3) return 'Maha (September - March)';
    if (month >= 5 && month <= 8) return 'Yala (May - August)';
    return 'Inter-season (April)';
  }

  List<SeasonalMarketAdvisory> advisoriesForDate(DateTime date) {
    final wom = weekOfMonth(date);
    return _advisories
        .where((item) => item.month == date.month && item.weekOfMonth == wom)
        .toList(growable: false);
  }
}
