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
  var value = Math.round(255 * currentText * 10);

  // Color on a spectrum between two defined colors
  let color_theirs = [255, 0, 0]; // Red #FF0000 rgb(255, 0, 0)
  let color_none = [0, 0, 0]; // Blue #45FE98 rgb(69, 254, 152)
  let color_ours = [69, 254, 152]; // Green

  let color_interp = value > 0 ? color_ours : color_theirs;
  value = value > 0 ? value : -value;

  // Calculate the color at the current value with log transformation to make the color spectrum more visible
  var color =
    "rgb(" +
    Math.round((value * (color_interp[0] - color_none[0])) / 255 + color_none[0]) +
    "," +
    Math.round((value * (color_interp[1] - color_none[1])) / 255 + color_none[1]) +
    "," +
    Math.round((value * (color_interp[2] - color_none[2])) / 255 + color_none[2]) +
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
