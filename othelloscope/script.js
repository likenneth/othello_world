// Get all table cells
var allTableCells = document.getElementsByTagName("td");

// Filter for the ones that have a title attribute
allTableCells = Array.prototype.filter.call(allTableCells, function (node) {
  return node.hasAttribute("title");
});

for (var i = 0, max = allTableCells.length; i < max; i++) {
  var node = allTableCells[i];

  // Get the title property from the cell
  var title = node.getAttribute("title");
  var currentText = title;

  // Convert to number
  currentText = Number(currentText);

  // Remap currentText from [log(0), log(1)] to [0, 255]
  currentText = Math.round(255 * currentText * 10);

  // Color on a spectrum between two defined colors
  var color1 = [0, 0, 0]; // Blue #45FE98 rgb(69, 254, 152)
  var color2 = [69, 254, 152]; // Green

  // Calculate the color at the current value with log transformation to make the color spectrum more visible
  var color =
    "rgb(" +
    Math.round((currentText * (color2[0] - color1[0])) / 255 + color1[0]) +
    "," +
    Math.round((currentText * (color2[1] - color1[1])) / 255 + color1[1]) +
    "," +
    Math.round((currentText * (color2[2] - color1[2])) / 255 + color1[2]) +
    ")";

  // Set the background color of the cell
  node.style.backgroundColor = color;
}

document.addEventListener("DOMContentLoaded", function () {
  // Get all td elements
  const tableCells = document.getElementsByTagName("td");

  for (let cell of tableCells) {
    // Add event listener for mouseenter
    cell.addEventListener("mouseenter", function () {
      cell.classList.add("expanded-cell");
    });

    // Add event listener for mouseleave
    cell.addEventListener("mouseleave", function () {
      cell.classList.remove("expanded-cell");
    });
  }
});
