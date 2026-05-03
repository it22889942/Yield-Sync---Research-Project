import 'package:flutter/material.dart';

class Nav {
  Nav._();

  static Future<void> go(BuildContext context, String route) {
    return Navigator.of(context).pushNamed(route);
  }

  static Future<void> replace(BuildContext context, String route) {
    return Navigator.of(context).pushReplacementNamed(route);
  }

  static Future<void> clearAndGo(BuildContext context, String route) {
    return Navigator.of(context).pushNamedAndRemoveUntil(route, (r) => false);
  }

  static void back(BuildContext context) => Navigator.of(context).pop();
}
