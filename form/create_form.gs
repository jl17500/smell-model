/**
 * Builds the "Smell ratings (v2)" Google Form plus a linked response sheet.
 *
 * Run once: go to https://script.google.com -> New project -> replace the code with this
 * file -> select createSmellForm -> Run -> allow access. The share link, edit link and
 * response-sheet link are printed in the Execution log.
 *
 * The model reads responses by these exact titles. Keep them in sync with
 * smell/taxonomy.py (QUIZ_NODES) and smell/data.py (RATING_QUESTION, FAVORITE_QUESTION),
 * and keep product names identical to data/real/catalog.csv.
 */
const SMELLS = [
  'Fresh roses',
  'Christmas tree (Evergreen)',
  'Air outside after summer rain',
  'Freshly baked cinnamon rolls',
  'Vanilla candle',
  'Freshly sliced ginger',
  'Fresh peaches',
  'Clean laundry',
  'Orange peel',
];

const PRODUCTS = [
  // Colognes / perfumes
  'Dior Sauvage',
  'Bleu de Chanel',
  'Giorgio Armani Acqua di Giò',
  'Versace Eros',
  'Lancôme La Vie Est Belle',
  'Marc Jacobs Daisy',
  'Glossier You',
  'Ariana Grande Cloud',
  // Candles
  'Bath & Body Works Mahogany Teakwood candle',
  'Bath & Body Works Eucalyptus Spearmint candle',
  'Bath & Body Works Warm Vanilla Sugar candle',
  'Bath & Body Works Champagne Toast candle',
  'Bath & Body Works Leaves candle',
  'Yankee Candle Balsam & Cedar',
  'Yankee Candle Clean Cotton',
  'Capri Blue Volcano candle',
];

const RATING_QUESTION = 'Rate the products you have smelled';
const LOVED_QUESTION = 'Scented products you already love (optional)';

function createSmellForm() {
  const form = FormApp.create('Smell ratings (v2)');
  form.setDescription(
    "We're building an app that recommends perfumes, colognes and candles from the everyday " +
    'smells you like. It takes about 2 minutes. We do not collect names or emails.');
  form.setCollectEmail(false);
  form.setConfirmationMessage('Thanks! Your answers help us match people to scents they will love.');
  try {
    form.setRequireLogin(false);  // school/work accounts default to members-only; open it to everyone
  } catch (e) {
    // Personal Google accounts don't have this setting; nothing to change.
  }

  // Section 1: everyday smells (the only thing a real app user will be asked).
  form.addSectionHeaderItem()
    .setTitle('Everyday smells')
    .setHelpText('Imagine a perfume, cologne or candle inspired by each smell. How much would you like it?');
  SMELLS.forEach(function (smell) {
    form.addScaleItem()
      .setTitle(smell)
      .setBounds(1, 6)
      .setLabels('Strongly dislike', 'Strongly like')
      .setRequired(true);
  });

  // Section 2: the answer key we use to train and test the model.
  form.addPageBreakItem()
    .setTitle('Products you have smelled')
    .setHelpText('Rate only the products you have actually smelled (in a store, on someone, at home). ' +
                 'Leave the rest blank.');
  form.addGridItem()
    .setTitle(RATING_QUESTION)
    .setHelpText('1 = strongly dislike, 5 = love it')
    .setRows(PRODUCTS)
    .setColumns(['1', '2', '3', '4', '5'])
    .setRequired(false);

  // Section 3: optional extra clue.
  form.addPageBreakItem().setTitle('Anything you already love? (optional)');
  form.addParagraphTextItem()
    .setTitle(LOVED_QUESTION)
    .setHelpText('Brand + exact product name, one per line, up to 3. Example: Glossier You');

  const sheet = SpreadsheetApp.create('Smell ratings (v2) responses');
  form.setDestination(FormApp.DestinationType.SPREADSHEET, sheet.getId());

  Logger.log('Share this link:  ' + form.getPublishedUrl());
  Logger.log('Edit the form:    ' + form.getEditUrl());
  Logger.log('Responses sheet:  ' + sheet.getUrl());
}
